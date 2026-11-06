#!/usr/bin/env python3
"""One-shot public-browser acceptance on a disposable GitHub-hosted Linux job.

No saved profile, credentials, UA override, security flags, challenge bypass,
artifact upload or download retry. The original ZIP is downloaded only here.
"""
from __future__ import annotations

import asyncio
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit
import zipfile

ORIGIN = "https://demo.gtr-voice.com"
GUIDE = "https://github.com/ZehuaKcrissLi/GTR-Voice/blob/main/docs/reproduction.md"
ZIP_BYTES = 1_044_673_451
ZIP_SHA256 = "d2b1301acbdeac7ca8e3808eae0d0f6396b9bb6c5afe4c84585c22a307c826e0"


def public_url(value):
    parsed = urlsplit(value)
    return parsed.scheme + "://" + parsed.netloc + parsed.path


def error_text(error):
    # GitHub asset redirects may contain temporary signed query strings.
    return re.sub(r"https?://[^\s<>\"']+", lambda m: public_url(m.group()), str(error))[:1800]


def emit(event, **fields):
    # Keep detailed receipts below per-line log limits. The event remains a
    # complete JSON object, formatted across lines for check/final receipts.
    indent = 2 if event in {"check_result", "final_result"} else None
    print(json.dumps({"event": event, **fields}, ensure_ascii=False, indent=indent), flush=True)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def assemble_media_ranges(parts):
    """Join only captured browser bytes; reject inconsistent totals/overlaps."""
    if not parts:
        return None, {"captured_ranges": [], "covered_bytes": 0, "total_bytes": None}
    totals = {part["total"] for part in parts}
    require(len(totals) == 1, "Media responses disagree about total byte length")
    total = totals.pop()
    require(0 < total <= 32 * 1024 * 1024, "Unexpected media resource byte length")
    runs = []
    for part in sorted(parts, key=lambda item: item["start"]):
        start, raw = part["start"], part["body"]
        require(raw and 0 <= start < total and start + len(raw) <= total and
                part["end"] == start + len(raw) - 1, "Invalid captured media interval")
        if not runs or start > runs[-1][0] + len(runs[-1][1]):
            runs.append((start, bytearray(raw)))
            continue
        run_start, buffer = runs[-1]
        offset = start - run_start
        overlap = min(len(raw), len(buffer) - offset)
        require(buffer[offset:offset + overlap] == raw[:overlap],
                "Overlapping browser media response bytes disagree")
        buffer.extend(raw[overlap:])
    evidence = {"total_bytes": total, "covered_bytes": sum(len(raw) for _, raw in runs),
                "captured_ranges": [{key: part[key] for key in ("start", "end", "total", "status")}
                                    for part in parts], "overlap_consistency_verified": True}
    complete = len(runs) == 1 and runs[0][0] == 0 and len(runs[0][1]) == total
    return bytes(runs[0][1]) if complete else None, evidence


class BrowserMediaCapture:
    """Observe real media responses from BEFORE navigation, including preload."""
    def __init__(self, page):
        self.page, self.parts, self.errors, self.tasks = page, {}, {}, set()
        self.pending, self.fatal, self.completions = {}, {}, {}
        self.closed = False
        page.on("response", self.observed)
        page.on("requestfinished", self.finished)
        page.on("requestfailed", self.failed)

    def completion(self, request):
        if request not in self.completions:
            self.completions[request] = asyncio.get_running_loop().create_future()
        return self.completions[request]

    def finish_request(self, request, error=None):
        parsed = urlsplit(request.url)
        if (self.closed or parsed.scheme + "://" + parsed.netloc != ORIGIN or
                not parsed.path.lower().endswith(".wav") or request.resource_type != "media"):
            return
        future = self.completion(request)
        if not future.done():
            # A result, rather than an exception, cannot become an un-retrieved
            # task exception when a metadata request is aborted or page closes.
            future.set_result(error)

    def finished(self, request):
        self.finish_request(request)

    def failed(self, request):
        self.finish_request(request, request.failure or "Browser requestfailed")

    def observed(self, response):
        parsed = urlsplit(response.url)
        if (self.closed or parsed.scheme + "://" + parsed.netloc != ORIGIN or
                not parsed.path.lower().endswith(".wav") or response.request.resource_type != "media"):
            return
        task = asyncio.create_task(self.capture(response))
        self.tasks.add(task)
        self.pending.setdefault(response.url, set()).add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(self.pending[response.url].discard)

    async def capture(self, response):
        url = response.url
        try:
            require(response.status in (200, 206), f"PUBLIC_AUDIO_MEDIA_HTTP_{response.status}")
            # Public requestfinished/requestfailed events provide the completion
            # witness. Do not cancel Response.finished(): Playwright creates an
            # inner task there which can report un-retrieved Target closed errors.
            completion = await asyncio.wait_for(asyncio.shield(self.completion(response.request)), timeout=25)
            if completion is not None:
                raise RuntimeError("Browser media response did not finish successfully: " + str(completion))
            raw = await asyncio.wait_for(response.body(), timeout=15)
            declared = response.headers.get("content-length")
            require(not declared or int(declared) == len(raw), "Captured media body length differs from header")
            if response.status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("content-range", ""))
                require(match is not None, "Missing/invalid media Content-Range")
                start, end, total = map(int, match.groups())
                require(0 <= start <= end < total and len(raw) == end - start + 1,
                        "Captured media bytes do not match declared range")
            else:
                require(not response.headers.get("content-range"), "HTTP 200 unexpectedly declares a partial range")
                start, end, total = 0, len(raw) - 1, len(raw)
            require(0 < total <= 32 * 1024 * 1024, "Unexpected media resource byte length")
            self.parts.setdefault(url, []).append({"start": start, "end": end, "total": total,
                                                   "status": response.status, "body": raw})
        except asyncio.CancelledError:
            raise
        except AssertionError as error:
            self.fatal.setdefault(url, []).append(error_text(error))
            self.errors.setdefault(url, []).append(error_text(error))
        except Exception as error:
            self.errors.setdefault(url, []).append(error_text(error))

    def available(self, url):
        require(not self.fatal.get(url), "Invalid observed browser media response: " + json.dumps(self.fatal.get(url, [])))
        return assemble_media_ranges(self.parts.get(url, []))

    async def settle_preload(self, url):
        # Do not abort an already observed metadata response by immediately
        # calling load(). Retain its prefix before permitting a browser reload.
        pending = list(self.pending.get(url, []))
        if pending:
            await asyncio.wait(pending, timeout=8)

    async def complete(self, url, seconds=30):
        deadline = asyncio.get_running_loop().time() + seconds
        version, result = -1, (None, {})
        while asyncio.get_running_loop().time() < deadline:
            require(not self.fatal.get(url), "Invalid observed browser media response: " + json.dumps(self.fatal.get(url, [])))
            parts = self.parts.get(url, [])
            if len(parts) != version:
                version = len(parts)
                result = assemble_media_ranges(parts)
                if result[0] is not None:
                    return result
            await asyncio.sleep(.1)
        raise AssertionError("Incomplete observed browser media coverage: " + json.dumps({
            **result[1], "capture_errors": self.errors.get(url, [])}))

    async def close(self):
        self.closed = True
        self.page.remove_listener("response", self.observed)
        self.page.remove_listener("requestfinished", self.finished)
        self.page.remove_listener("requestfailed", self.failed)
        for future in self.completions.values():
            if not future.done():
                future.set_result("Media capture closed")
        pending = list(self.tasks)
        if pending:
            _, pending = await asyncio.wait(pending, timeout=2)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def challenge_check(page):
    title = await page.title()
    blocked = await page.locator(
        "#challenge-form, #challenge-running, iframe[src*='challenges.cloudflare.com']"
    ).count()
    text = (await page.locator("body").inner_text())[:6000]
    require(not blocked and not re.search(
        r"just a moment|verify you are human|checking your browser|attention required.*cloudflare",
        title + "\n" + text, re.I), "PUBLIC_BROWSER_CHALLENGE: no bypass attempted")


async def navigate(page, path):
    response = await page.goto(ORIGIN + path, wait_until="domcontentloaded", timeout=35_000)
    await challenge_check(page)
    require(response is not None and response.status == 200,
            f"Public document HTTP {response.status if response is not None else chr(63)}; expected 200")
    require(urlsplit(page.url).netloc == urlsplit(ORIGIN).netloc,
            "Unexpected origin document redirect")


async def media_probe(page, audio, *, already_playing=False, test_drag=False):
    """Actual native-control clicks; no play()/currentTime assignment shortcut."""
    await audio.scroll_into_view_if_needed()
    source = await audio.evaluate("a => a.currentSrc || a.src")
    require(public_url(source).startswith(ORIGIN + "/"), "Audio left the public origin")
    if not already_playing:
        box = await audio.bounding_box()
        require(box and box["width"] >= 180, "Native audio controls are too narrow")
        await audio.click(position={"x": 18, "y": box["height"] / 2})
    handle = await audio.element_handle()
    await page.wait_for_function(
        "a => !a.paused && Number.isFinite(a.duration) && a.currentTime > 0.2",
        arg=handle, timeout=15_000)
    first = await audio.evaluate("a => ({time:a.currentTime,duration:a.duration})")
    await page.wait_for_timeout(350)
    second = await audio.evaluate("a => a.currentTime")
    require(second > first["time"] + .08, "Media clock did not advance during playback")
    await audio.evaluate("a => a.pause()")
    before = await audio.evaluate("a => a.currentTime")
    await audio.evaluate("""a => {
      a.__publicVerifierSeeks=0;
      a.addEventListener('seeked',()=>{a.__publicVerifierSeeks++;});
    }""")
    box = await audio.bounding_box()
    sought, seek_fraction = None, None
    # Chromium's native timeline is centered; these are real pointer clicks.
    # Try other positions only if a click did not seek, never a JS assignment.
    for fraction in (.55, .65, .45):
        await audio.click(position={"x": box["width"] * fraction,
                                    "y": box["height"] / 2})
        await page.wait_for_timeout(250)
        after = await audio.evaluate("a => ({time:a.currentTime,seeks:a.__publicVerifierSeeks})")
        if after["seeks"] > 0 and abs(after["time"] - before) > min(.5, first["duration"] * .12):
            sought = after["time"]
            seek_fraction = fraction
            break
    await audio.evaluate("a => a.pause()")
    require(sought is not None, "Native timeline click did not produce an observable seek")
    result = {"path": urlsplit(source).path, "duration": first["duration"],
            "playback_clock_advanced": True, "seek_before": before, "seek_after": sought,
            "seek_method": "native_control_pointer_click", "physical_drag_requested": test_drag,
            "physical_drag_verified": False}
    if test_drag:
        try:
            # The successful timeline click identifies a real native slider
            # position. Start there, then send a genuine down/move/up sequence.
            box = await audio.bounding_box()
            start_x = box["x"] + box["width"] * seek_fraction
            y = box["y"] + box["height"] / 2
            delta = min(32, box["width"] * .08)
            await page.mouse.move(start_x, y)
            await page.mouse.down()
            try:
                # A track press can itself seek. Exclude that change from the
                # drag proof: only measure movement/release after mouse-down.
                await page.wait_for_timeout(120)
                state = await audio.evaluate("a => ({time:a.currentTime,seeks:a.__publicVerifierSeeks})")
                await page.mouse.move(start_x + delta, y, steps=8)
            finally:
                await page.mouse.up()
            await page.wait_for_timeout(350)
            after_drag = await audio.evaluate("a => ({time:a.currentTime,seeks:a.__publicVerifierSeeks})")
            await audio.evaluate("a => a.pause()")
            result["drag"] = {"method": "native_pointer_down_move_up", "horizontal_pixels": delta,
                              "before": state["time"], "after": after_drag["time"],
                              "new_seeked_events": after_drag["seeks"] - state["seeks"]}
            require(after_drag["seeks"] > state["seeks"] and
                    abs(after_drag["time"] - state["time"]) > min(.25, first["duration"] * .06),
                    "Physical native timeline drag did not produce a new seeked event and time change")
            result["physical_drag_verified"] = True
        except Exception as error:
            result["drag_error"] = error_text(error)
    return result


async def homepage(context):
    page = await context.new_page()
    try:
        await navigate(page, "/")
        text = await page.locator("body").inner_text()
        require(not re.search(r"Local recovery preview|\bPARTIAL\b|Dataset public download is not enabled",
                              text, re.I), "Internal recovery banner is publicly visible")
        link = page.get_by_role("link", name="Reproduction guide", exact=True)
        require(await link.count() == 1, "Reproduction guide link is not unique")
        require(await link.get_attribute("href") == GUIDE, "Guide href differs from the approved URL")
        async with page.expect_popup(timeout=30_000) as opened:
            await link.click()
        guide = await opened.value
        try:
            await guide.wait_for_load_state("domcontentloaded", timeout=35_000)
            await challenge_check(guide)
            require(guide.url == GUIDE, "Guide popup did not reach the exact public document")
            await guide.get_by_role("heading", name="Reproducing GTR-Voice research paths",
                                    exact=True).wait_for(timeout=30_000)
            require("Measured runs on the original released audio" in await guide.locator("body").inner_text(),
                    "Guide popup did not render the reproduction content")
        finally:
            await guide.close()
        return {"passed": True, "no_internal_banner": True,
                "guide_unique_exact_href_and_real_popup_content": True}
    finally:
        await page.close()


async def comparisons(context, network):
    page = await context.new_page()
    try:
        await navigate(page, "/")
        await page.wait_for_function("document.querySelectorAll('.audio-table tbody audio').length === 36")
        require(await page.locator(".audio-table tbody tr").count() == 12, "Comparison rows differ from 12")
        results, failures = [], []
        for index in range(36):
            try:
                result = await media_probe(page, page.locator(".audio-table tbody audio").nth(index),
                                           test_drag=index == 0)
                result["player_index"] = index
                result["same_origin_media_http_206"] = any(
                    item["path"] == result["path"] and item["status"] == 206 for item in network)
                require(result["same_origin_media_http_206"], "No observed browser media HTTP 206")
                results.append(result)
                if index == 0 and not result["physical_drag_verified"]:
                    failures.append({"player_index": index, "check": "physical_drag",
                                     "error": result.get("drag_error", "Drag not verified")})
            except Exception as error:
                failures.append({"player_index": index, "error": error_text(error)})
            emit("comparison_progress", completed=index + 1, failures=len(failures))
        return {"passed": not failures, "rows": 12, "players": 36,
                "verified_play_and_seek": results, "failures": failures}
    finally:
        await page.close()


async def graph(context):
    page = await context.new_page()
    diagnostics, page_errors, player = [], [], None
    clicked, attempts, points = False, 0, None
    page.on("pageerror", lambda error: page_errors.append(error_text(error)) if len(page_errors) < 20 else None)
    async def snapshot():
        return await page.evaluate("""() => {
          const a=document.querySelector('#gtr-graph-audio');
          return {hover:window.__publicVerifierHover || null,
            observed_click:window.__publicVerifierClick || null,
            player:a ? {src:a.src,currentSrc:a.currentSrc,paused:a.paused,
              currentTime:a.currentTime,duration:Number.isFinite(a.duration)?a.duration:null,
              readyState:a.readyState,networkState:a.networkState,
              error:a.error ? {code:a.error.code,message:a.error.message}:null}:null};
        }""")
    try:
        await navigate(page, "/GTR-3D-plot")
        await page.wait_for_function("""() => {
          const p=document.querySelector('.plotly-graph-div');
          return p && p.data && p._fullLayout && p.querySelector('canvas');
        }""", timeout=30_000)
        points = await page.locator(".plotly-graph-div").evaluate(
            "p => p.data.filter(t=>t.type==='scatter3d').reduce((n,t)=>n+t.x.length,0)")
        require(points == 125, "Rendered graph data is not the original 125 points")
        await page.locator(".plotly-graph-div").evaluate("""p => {
          window.__publicVerifierHover=null;
          window.__publicVerifierClick={sequence:0,src:null};
          p.on('plotly_hover', e => { window.__publicVerifierHover=e.points[0].customdata[0]; });
          p.on('plotly_unhover', () => { window.__publicVerifierHover=null; });
          p.on('plotly_click', e => { window.__publicVerifierClick={
            sequence:window.__publicVerifierClick.sequence+1,
            src:new URL(e.points[0].customdata[0],location.href).href}; });
        }""")
        box = await page.locator(".plotly-graph-div").bounding_box()
        attempted_sources = set()
        # Listen to real Plotly hover events while moving the real pointer.
        # Do not emit a plotly_click or assign the player's source to fake a hit.
        for y in range(70, min(int(box["height"]) - 30, 470), 22):
            for x in range(100, int(box["width"]) - 130, 22):
                await page.mouse.move(box["x"] + x, box["y"] + y)
                await page.wait_for_timeout(25)
                hovered = await page.evaluate("window.__publicVerifierHover")
                if hovered and not hovered.endswith("/010/18.wav") and hovered not in attempted_sources:
                    # Allow the genuine 3D hit-test to settle, then send an
                    # actual down/up pair. Never immediately toggle the same
                    # playing point by repeatedly clicking its source.
                    await page.wait_for_timeout(100)
                    settled = await page.evaluate("window.__publicVerifierHover")
                    if settled != hovered:
                        continue
                    attempted_sources.add(hovered)
                    before_click = await page.evaluate("window.__publicVerifierClick.sequence")
                    before_state = await snapshot()
                    await page.mouse.click(box["x"] + x, box["y"] + y, delay=80)
                    attempts += 1
                    click_error = None
                    try:
                        await page.wait_for_function("""previous => {
                          const click=window.__publicVerifierClick;
                          const audio=document.querySelector('#gtr-graph-audio');
                          return click.sequence>previous && click.src &&
                            !click.src.endsWith('/010/18.wav') && audio.src===click.src &&
                            audio.currentSrc===click.src && !audio.paused && audio.currentTime>0.1;
                        }""", arg=before_click, timeout=3000)
                        clicked = True
                    except Exception as error:
                        click_error = error_text(error)
                    diagnostics.append({"attempt":attempts,"hover_url":hovered,
                                        "position":{"x":x,"y":y},"before":before_state,
                                        "after":await snapshot(),"wait_error":click_error})
                    if clicked:
                        break
                if attempts >= 8:
                    break
            if clicked or attempts >= 8:
                break
        require(clicked, "GRAPH_REAL_CLICK_PLAYBACK_NOT_VERIFIED: inspect actual event/player diagnostics; no bypass attempted")
        player = await media_probe(page, page.locator("#gtr-graph-audio"), already_playing=True, test_drag=True)
        return {"passed": player["physical_drag_verified"], "scatter3d_points": points,
                "point_click": "real_pointer_with_observed_plotly_click_and_matching_playing_src",
                "physical_click_attempts": attempts,
                "observed_click": await page.evaluate("window.__publicVerifierClick"),
                "attempt_diagnostics":diagnostics,"page_errors":page_errors,
                "player": player, "physical_drag_verified": player["physical_drag_verified"]}
    except Exception as error:
        try:
            last_state = await snapshot()
        except Exception as state_error:
            last_state = {"snapshot_error":error_text(state_error)}
        return {"passed":False,"error":error_text(error),"scatter3d_points":points,
                "physical_click_attempts":attempts,"real_click_and_matching_playback_verified":clicked,
                "attempt_diagnostics":diagnostics,"last_state":last_state,
                "page_errors":page_errors,"player":player,
                "physical_drag_verified":bool(player and player.get("physical_drag_verified"))}
    finally:
        await page.close()


async def evaluation(context, path):
    page = await context.new_page()
    capture = BrowserMediaCapture(page)  # installed before navigation/autopreload
    try:
        await navigate(page, path)
        require(await page.locator("table audio").count() == 72, "Evaluation does not contain 72 stimuli")
        sources = await page.locator("table audio").evaluate_all("items=>items.map(a=>a.src)")
        require(all(public_url(url).startswith(ORIGIN + "/") for url in sources), "Evaluation audio left origin")
        # Retain initial browser metadata prefixes and subsequent real ranges.
        # A cached complete capture needs no new response. No API, cache-busting
        # URL, fabricated range request, CSP override or altered UA is used.
        semaphore = asyncio.Semaphore(4)
        async def decode_public_resource(index, url):
            async with semaphore:
                try:
                    audio = page.locator("table audio").nth(index)
                    await capture.settle_preload(url)
                    raw, coverage = capture.available(url)
                    if raw is None:
                        await audio.evaluate("a => { a.preload='auto'; a.load(); }")
                        raw, coverage = await capture.complete(url)
                    require(raw[:4] == b"RIFF" and raw[8:12] == b"WAVE",
                            "PUBLIC_AUDIO_NOT_WAV_OR_CHALLENGE; no bypass attempted")
                    decoded_audio = await page.evaluate("""async encoded => {
                      const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
                      const ctx=new AudioContext();
                      try {
                        const audio=await ctx.decodeAudioData(bytes.buffer);
                        if(!audio.length || !Number.isFinite(audio.duration) || audio.duration<=0) throw Error('Empty/nonfinite decode');
                        for(let c=0;c<audio.numberOfChannels;c++) {
                          const channel=audio.getChannelData(c);
                          for(let i=0;i<channel.length;i++) if(!Number.isFinite(channel[i])) throw Error('Nonfinite PCM');
                        }
                        return {frames:audio.length,decoded_sample_rate:audio.sampleRate};
                      } finally { await ctx.close(); }
                    }""", base64.b64encode(raw).decode("ascii"))
                    return {"path": urlsplit(url).path, "passed": True,
                            "bytes": len(raw), "browser_range_coverage": coverage,
                            "other_capture_errors": capture.errors.get(url, []),
                            "complete_body_verified": True, **decoded_audio}
                except Exception as error:
                    return {"path": urlsplit(url).path, "passed": False, "error": error_text(error)}
        decoded = await asyncio.gather(*(decode_public_resource(index, url) for index, url in enumerate(sources)))
        player, player_error = None, None
        try:
            player = await media_probe(page, page.locator("table audio").first, test_drag=True)
        except Exception as error:
            player_error = error_text(error)
        failures = [item for item in decoded if not item["passed"]]
        return {"passed": not failures and player_error is None and bool(player and player["physical_drag_verified"]), "page": path,
                "stimuli": len(sources), "browser_decoded": sum(item["passed"] for item in decoded),
                "resource_transport": "actual browser preload/load response ranges, contiguous complete bytes; not fetch/API client",
                "decoder": "Chromium AudioContext.decodeAudioData", "page_csp_unchanged": True,
                "decoded": decoded, "player": player, "player_error": player_error}
    finally:
        await capture.close()
        await page.close()


async def dataset_download(context, scratch):
    page = await context.new_page()
    try:
        await navigate(page, "/")
        link = page.get_by_role("link", name="Dataset Download", exact=True)
        require(await link.count() == 1, "Dataset Download is not a unique visitor link")
        require(await link.evaluate("a=>a.href") == ORIGIN + "/GTR-Voice.zip", "Dataset link changed unexpectedly")
        emit("full_download_started", expected_bytes=ZIP_BYTES, attempts=1)
        async with page.expect_download(timeout=45_000) as started:
            await link.click()
        download = await started.value
        require(download.suggested_filename == "GTR-Voice.zip", "Unexpected downloaded filename")
        output = scratch / "complete-original.zip"
        try:
            await asyncio.wait_for(download.save_as(output), timeout=300)
        except Exception:
            await download.cancel()
            raise
        require(await download.failure() is None, "Browser download reported failure")
        require(output.stat().st_size == ZIP_BYTES, "Complete ZIP byte length differs")
        digest = hashlib.sha256()
        with output.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        require(digest.hexdigest() == ZIP_SHA256, "Complete ZIP SHA256 differs")
        with zipfile.ZipFile(output) as archive:
            entries = archive.infolist()
            wavs = [entry for entry in entries if not entry.is_dir() and entry.filename.lower().endswith(".wav")]
            require(len(wavs) == 2500 and len({entry.filename for entry in wavs}) == 2500,
                    "ZIP does not contain 2500 unique WAV file members")
        return {"passed": True, "method": "real_link_click_expect_download_save_as_full_file_hash",
                "download_attempts": 1, "bytes": output.stat().st_size, "sha256": digest.hexdigest(),
                "zip_entries_including_directories": len(entries), "zip_wav_file_members": len(wavs),
                "crc_test": "not_run; exact whole-file SHA256 verified", "extracted": False}
    finally:
        await page.close()


async def verify(scratch, report, *, media_only=False):
    from playwright.async_api import async_playwright
    network, legacy = [], []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            channel="chrome", headless=True, chromium_sandbox=True,
            downloads_path=str(scratch / "browser-downloads"))
        report["browser"] = {"engine": "preinstalled Google Chrome (Chromium)", "version": browser.version,
                             "profile": "fresh_nonpersistent", "sandbox_enabled": True,
                             "user_agent_override": False, "closed": False}
        try:
            context = await browser.new_context(accept_downloads=True, viewport={"width": 1440, "height": 1000})
            context.set_default_timeout(20_000)
            def observed(response):
                url = urlsplit(response.url)
                if url.scheme + "://" + url.netloc == ORIGIN and url.path.lower().endswith(".wav"):
                    network.append({"path": url.path, "status": response.status,
                                    "content_range": response.headers.get("content-range")})
            def requested(request):
                host = urlsplit(request.url).netloc.lower()
                if "ngrok" in host:
                    legacy.append(public_url(request.url))
            context.on("response", observed)
            context.on("request", requested)
            # Complete the single ZIP attempt early, so a later WebGL/media
            # environment failure cannot discard download evidence.
            checks = [("homepage_and_guide", 90, lambda: homepage(context)),
                      ("comparison_audio", 180, lambda: comparisons(context, network)),
                      ("graph_click_and_timeline", 60, lambda: graph(context)),
                      ("fastpitch_evaluation", 120, lambda: evaluation(context, "/fspt/evaluation_1.html")),
                      ("styletts_evaluation", 120, lambda: evaluation(context, "/stts/evaluation_22.html"))]
            if media_only:
                report["checks"]["complete_dataset_download"] = {
                    "status": "SKIPPED_MEDIA_ONLY", "passed": None, "download_attempts": 0,
                    "reason": "This run supplies no ZIP download proof; retain separate previously verified evidence"}
                emit("check_skipped", check="complete_dataset_download", reason="media_only")
            else:
                checks.insert(1, ("complete_dataset_download", 420, lambda: dataset_download(context, scratch)))
            for name, seconds, action in checks:
                emit("check_started", check=name)
                try:
                    result = await asyncio.wait_for(action(), timeout=seconds)
                except Exception as error:
                    result = {"passed": False, "error": error_text(error) or type(error).__name__}
                report["checks"][name] = result
                emit("check_finished", check=name, passed=result["passed"])
                emit("check_result", check=name, result=result)
            report["network"] = {"observed_same_origin_audio_http_206": sum(item["status"] == 206 for item in network),
                                 "audio_206_examples": [item for item in network if item["status"] == 206][:5],
                                 "legacy_ngrok_requests": legacy}
            require(not legacy, "Observed requests to a legacy ngrok host")
            require(report["network"]["observed_same_origin_audio_http_206"] > 0, "No actual browser media HTTP 206 observed")
            await context.close()
        finally:
            await browser.close()
            report["browser"]["closed"] = True


def finalize_report(report, *, media_only):
    """Pure, None-safe terminal accounting, also used by synthetic fixtures."""
    def check(name):
        value = report["checks"].get(name)
        return value if isinstance(value, dict) else {}
    def player(name):
        value = check(name).get("player")
        return value if isinstance(value, dict) else {}
    expected = {"homepage_and_guide", "comparison_audio", "graph_click_and_timeline",
                "fastpitch_evaluation", "styletts_evaluation"}
    if not media_only:
        expected.add("complete_dataset_download")
    for name in sorted(expected):
        if not check(name).get("passed"):
            report["failures"].append(name)
    first = next((item for item in (check("comparison_audio").get("verified_play_and_seek") or [])
                  if isinstance(item, dict) and item.get("player_index") == 0), {})
    report["drag_checks"] = {
        "homepage_first_player": first.get("physical_drag_verified", False),
        "graph_player": player("graph_click_and_timeline").get("physical_drag_verified", False),
        "fastpitch_evaluation": player("fastpitch_evaluation").get("physical_drag_verified", False),
        "styletts_evaluation": player("styletts_evaluation").get("physical_drag_verified", False),
    }
    report["physical_drag_verified"] = all(report["drag_checks"].values())
    report["complete_dataset_download_verified_this_run"] = bool(
        not media_only and check("complete_dataset_download").get("passed"))
    success = not report["failures"] and report.get("cleanup", {}).get("temporary_downloads_deleted")
    report["status"] = ("PASS_MEDIA_ONLY" if media_only else "PASS") if success else "FAILED_OR_PARTIAL"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-only", action="store_true",
                        help="Skip ZIP transfer explicitly; success is PASS_MEDIA_ONLY, never a full-download PASS")
    args = parser.parse_args()
    report = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "origin": ORIGIN, "scope": "fresh disposable Chromium; public resources only",
              "checks": {}, "failures": [], "physical_drag_verified": False,
              "verification_mode": "media_only" if args.media_only else "full",
              "human_listening_quality_rated": False}
    temporary = None
    try:
        require(os.environ.get("GITHUB_ACTIONS") == "true" and platform.system() == "Linux",
                "REFUSED_NOT_GITHUB_LINUX_RUNNER: no local ZIP download is permitted")
        launcher = shutil.which("google-chrome")
        require(launcher and Path("/opt/google/chrome/chrome").is_file(),
                "Required preinstalled system Google Chrome is unavailable; no binary fallback")
        version = subprocess.run([launcher, "--version"], check=True, capture_output=True,
                                 text=True, timeout=10).stdout.strip()
        report["platform"] = {"system": platform.system(), "release": platform.release(),
                              "machine": platform.machine(), "google_chrome_launcher": launcher,
                              "google_chrome_binary": "/opt/google/chrome/chrome",
                              "google_chrome_version": version, "binary_fallback_permitted": False}
        emit("platform_preflight", **report["platform"])
        temporary = tempfile.TemporaryDirectory(prefix="gtr-public-browser-")
        scratch = Path(temporary.name)
        minimum = 256 * 1024 ** 2 if args.media_only else 3 * ZIP_BYTES + 1024 ** 3
        require(shutil.disk_usage(scratch).free >= minimum,
                "Insufficient ephemeral runner space for browser and saved ZIP")
        emit("ephemeral_storage_created", minimum_free_bytes=minimum)
        async def bounded():
            await asyncio.wait_for(verify(scratch, report, media_only=args.media_only), timeout=1000)
        asyncio.run(bounded())
    except Exception as error:
        report["failures"].append(error_text(error))
    finally:
        if temporary is not None:
            root = Path(temporary.name)
            try:
                retained = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
            except OSError:
                retained = None
            try:
                temporary.cleanup()
                report["cleanup"] = {"temporary_downloads_deleted": not root.exists(),
                                     "temporary_file_bytes_before_removal": retained,
                                     "browser_context": "closed on normal completion; isolated runner lifetime bounds failures"}
            except Exception as error:
                report["cleanup"] = {"temporary_downloads_deleted": False, "error": error_text(error)}
                report["failures"].append("Temporary download cleanup failed")
            emit("cleanup", **report["cleanup"])
    try:
        finalize_report(report, media_only=args.media_only)
    except Exception as error:
        report["failures"].append("Terminal accounting error: " + error_text(error))
        report["status"] = "FAILED_OR_PARTIAL"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write("## GTR public browser verification: " + report["status"] + "\n\n```json\n")
                stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n")
        except Exception as error:
            report["failures"].append("Step summary write failed: " + error_text(error))
            report["status"] = "FAILED_OR_PARTIAL"
    # Always print the terminal JSON, including failures, after cleanup and
    # accounting. Pretty lines avoid losing a >64 KiB single log record.
    emit("final_result", result=report)
    return 0 if report["status"] in {"PASS", "PASS_MEDIA_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
