# Fastapi imports
from typing import Union
from fastapi import FastAPI


@app.get("/query/")
async def getTopResults(artist: str, track: str, top: int):
    return [ artist, track ,top]