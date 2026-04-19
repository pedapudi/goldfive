"""Unit tests for the ControlChannel primitive (goldfive/control.py).

These tests pin the contract that Agent B (executor integration) and Agent E
(harmonograf bridge) will build against. They do NOT exercise any runner
behavior — pure send/receive/ack round-trip and lifecycle.
"""

from __future__ import annotations

import asyncio

import pytest

from goldfive.control import (
    AckResult,
    ControlAck,
    ControlChannel,
    ControlKind,
    ControlMessage,
)


@pytest.mark.asyncio
async def test_send_receive_round_trip() -> None:
    channel = ControlChannel()
    msg = ControlMessage(kind=ControlKind.STEER, payload={"note": "focus on slide 5"})

    await channel.send(msg)
    received = await channel.receive(timeout=0.1)

    assert received is msg
    assert received.kind == ControlKind.STEER
    assert received.payload == {"note": "focus on slide 5"}
    assert received.id  # auto-generated


@pytest.mark.asyncio
async def test_ack_round_trip() -> None:
    channel = ControlChannel()
    msg = ControlMessage(kind=ControlKind.PAUSE)

    await channel.send(msg)
    received = await channel.receive(timeout=0.1)
    assert received is not None

    ack = ControlAck(control_id=received.id, result=AckResult.SUCCESS, detail="paused")
    await channel.ack(ack)

    collected: list[ControlAck] = []

    async def collect() -> None:
        async for a in channel.acks():
            collected.append(a)

    task = asyncio.create_task(collect())
    # Yield so the consumer picks up the ack, then close to terminate the iterator.
    await asyncio.sleep(0.01)
    channel.close()
    await asyncio.wait_for(task, timeout=0.1)

    assert len(collected) == 1
    assert collected[0].control_id == received.id
    assert collected[0].result == AckResult.SUCCESS
    assert collected[0].detail == "paused"


@pytest.mark.asyncio
async def test_receive_timeout_returns_none() -> None:
    channel = ControlChannel()
    result = await channel.receive(timeout=0.05)
    assert result is None


@pytest.mark.asyncio
async def test_close_stops_receive_and_acks() -> None:
    channel = ControlChannel()

    # A pending acks() consumer should wake up and exit the iterator on close.
    seen: list[ControlAck] = []

    async def consume_acks() -> None:
        async for a in channel.acks():
            seen.append(a)

    consumer = asyncio.create_task(consume_acks())
    await asyncio.sleep(0.01)  # let consumer block on _outbox.get()

    channel.close()
    await asyncio.wait_for(consumer, timeout=0.1)
    assert seen == []

    # After close, receive() returns None immediately even with no timeout.
    result = await channel.receive()
    assert result is None


@pytest.mark.asyncio
async def test_multiple_messages_queue_in_order() -> None:
    channel = ControlChannel()
    kinds = [ControlKind.PAUSE, ControlKind.RESUME, ControlKind.CANCEL]
    for k in kinds:
        await channel.send(ControlMessage(kind=k))

    received_kinds = []
    for _ in kinds:
        msg = await channel.receive(timeout=0.1)
        assert msg is not None
        received_kinds.append(msg.kind)

    assert received_kinds == kinds
