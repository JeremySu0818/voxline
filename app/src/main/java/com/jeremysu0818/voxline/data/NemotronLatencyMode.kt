package com.jeremysu0818.voxline.data

enum class NemotronLatencyMode(
    val id: String,
    val latencyMs: Int,
    val rightContextFrames: Int,
) {
    ULTRA_LOW("80", 80, 0),
    VERY_LOW("160", 160, 1),
    BALANCED("320", 320, 3),
    ACCURATE("560", 560, 6),
    MOST_ACCURATE("1120", 1120, 13);

    val displayName: String
        get() = "$latencyMs ms"

    companion object {
        val default: NemotronLatencyMode = BALANCED

        fun fromId(id: String?): NemotronLatencyMode =
            entries.firstOrNull { it.id == id } ?: default
    }
}
