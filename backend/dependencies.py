"""
Shared component instances — accessed via app.state in route handlers.
These getter functions provide typed access for route handlers.
"""

from fastapi import Request


def get_loop(request: Request):
    return request.app.state.loop


def get_emitter(request: Request):
    return request.app.state.emitter


def get_store(request: Request):
    return request.app.state.store


def get_curriculum(request: Request):
    return request.app.state.curriculum
