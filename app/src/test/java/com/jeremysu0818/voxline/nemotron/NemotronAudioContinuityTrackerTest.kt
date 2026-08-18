package com.jeremysu0818.voxline.nemotron

import com.jeremysu0818.voxline.audio.CapturedAudioChunk
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test

class NemotronAudioContinuityTrackerTest {
    private val tracker = NemotronAudioContinuityTracker()

    @Test
    fun `continuous chunks preserve streaming state`() {
        assertSame(
            NemotronAudioContinuityTracker.Result.Continuous,
            tracker.observe(chunk(start = 0, size = 1280)),
        )
        assertSame(
            NemotronAudioContinuityTracker.Result.Continuous,
            tracker.observe(chunk(start = 1280, size = 1280)),
        )
    }

    @Test
    fun `gap reports exact dropped audio duration`() {
        tracker.observe(chunk(start = 0, size = 1280))
        val result = tracker.observe(chunk(start = 3840, size = 1280))

        result as NemotronAudioContinuityTracker.Result.Gap
        assertEquals(2560L, result.droppedSamples)
        assertEquals(160.0, result.droppedAudioMs, 0.0001)
    }

    @Test
    fun `stale chunks never move expected cursor backwards`() {
        tracker.observe(chunk(start = 0, size = 1280))
        assertSame(
            NemotronAudioContinuityTracker.Result.Stale,
            tracker.observe(chunk(start = 640, size = 1280)),
        )
        assertSame(
            NemotronAudioContinuityTracker.Result.Continuous,
            tracker.observe(chunk(start = 1280, size = 1280)),
        )
    }

    @Test
    fun `reset accepts a new absolute stream origin`() {
        tracker.observe(chunk(start = 10_000, size = 1280))
        tracker.reset()
        assertSame(
            NemotronAudioContinuityTracker.Result.Continuous,
            tracker.observe(chunk(start = 0, size = 1280)),
        )
    }

    private fun chunk(start: Long, size: Int) = CapturedAudioChunk(
        samples = ShortArray(size),
        startSample = start,
        capturedAtNanos = 0,
    )
}
