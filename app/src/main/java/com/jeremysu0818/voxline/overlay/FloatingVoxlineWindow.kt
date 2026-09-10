package com.jeremysu0818.voxline.overlay

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.PixelFormat
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.zIndex
import androidx.lifecycle.*
import androidx.savedstate.*
import com.jeremysu0818.voxline.data.VoxlineLine
import com.jeremysu0818.voxline.data.VoxlineRuntimeStore
import com.jeremysu0818.voxline.data.t
import com.jeremysu0818.voxline.ui.theme.VoxlineTheme
import kotlinx.coroutines.delay
import kotlin.math.abs
import kotlin.math.roundToInt

private class OverlayLifecycleOwner : LifecycleOwner, ViewModelStoreOwner, SavedStateRegistryOwner {
    private val lifecycleRegistry = LifecycleRegistry(this)
    private val savedStateRegistryController = SavedStateRegistryController.create(this)
    private val store = ViewModelStore()

    override val lifecycle: Lifecycle get() = lifecycleRegistry
    override val savedStateRegistry: SavedStateRegistry get() = savedStateRegistryController.savedStateRegistry
    override val viewModelStore: ViewModelStore get() = store

    fun init() {
        savedStateRegistryController.performRestore(null)
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_CREATE)
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_START)
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_RESUME)
    }

    fun destroy() {
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_PAUSE)
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_STOP)
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_DESTROY)
        store.clear()
    }
}

private const val OverlayWindowAlpha = 1f

class FloatingVoxlineState(
    initialX: Int,
    initialY: Int,
    val minHeightPx: Int,
    val maxHeightPx: Int,
) {
    var x by mutableIntStateOf(initialX)
    var y by mutableIntStateOf(initialY)
    var heightPx by mutableIntStateOf(minHeightPx)
}

data class WindowPosition(val x: Int, val y: Int)

private class CloseTargetState {
    var isVisible by mutableStateOf(false)
    var isActive by mutableStateOf(false)

    fun show() {
        isVisible = true
    }

    fun hide() {
        isVisible = false
        isActive = false
    }
}

class FloatingVoxlineWindow(
    private val context: Context,
    private val onCloseRequested: () -> Unit,
    private val windowType: Int = WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
) {
    private val mainHandler = Handler(Looper.getMainLooper())
    private val windowManager = context.getSystemService(WindowManager::class.java)

    private var mainView: ComposeView? = null
    private var closeTargetView: ComposeView? = null

    private var mainLifecycle: OverlayLifecycleOwner? = null
    private var closeTargetLifecycle: OverlayLifecycleOwner? = null

    @SuppressLint("ClickableViewAccessibility")
    fun show() {
        mainHandler.post {
            if (mainView != null) return@post

            val density = context.resources.displayMetrics.density
            val screenWidthPixels = context.resources.displayMetrics.widthPixels
            val screenHeightPixels = context.resources.displayMetrics.heightPixels
            val baseWidth = (screenWidthPixels - 32 * density).coerceAtMost(720 * density)
            val windowWidthPx = (baseWidth * 0.9f).roundToInt()
            val barHeightPx = (16 * density).roundToInt()
            val closeTargetHeightPx = (112 * density).roundToInt()
            val closeCapsuleWidthPx = (128 * density).roundToInt()
            val closeCapsuleHeightPx = (48 * density).roundToInt()
            val closeCapsuleBottomMarginPx = (20 * density).roundToInt()
            val minHeightPx = (screenHeightPixels / 6f).roundToInt()
            val maxHeightPx = (screenHeightPixels * 0.6f).roundToInt()
            val initialTopY = (screenHeightPixels * 0.72f).roundToInt()

            val state = FloatingVoxlineState(
                initialX = (screenWidthPixels - windowWidthPx) / 2,
                initialY = initialTopY,
                minHeightPx = minHeightPx,
                maxHeightPx = maxHeightPx,
            )

            val mainParams = WindowManager.LayoutParams(
                windowWidthPx,
                state.heightPx,
                windowType,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                    WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT,
            ).apply {
                gravity = Gravity.TOP or Gravity.START
                x = state.x
                y = state.y
                alpha = OverlayWindowAlpha
            }

            val mainOwner = OverlayLifecycleOwner().apply { init() }
            val closeTargetOwner = OverlayLifecycleOwner().apply { init() }
            val closeTargetState = CloseTargetState()

            val closeTargetParams = WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                closeTargetHeightPx,
                windowType,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT,
            ).apply {
                gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
                alpha = OverlayWindowAlpha
            }

            val captionView = ComposeView(context).apply {
                setViewTreeLifecycleOwner(mainOwner)
                setViewTreeViewModelStoreOwner(mainOwner)
                setViewTreeSavedStateRegistryOwner(mainOwner)
                setContent {
                    VoxlineTheme {
                        FloatingCaptionApp(
                            barHeightPx = barHeightPx,
                            onDragStarted = closeTargetState::show,
                            onMove = { dx, dy, touchX, touchY ->
                                state.x += dx.roundToInt()
                                state.y += dy.roundToInt()
                                mainParams.x = state.x
                                mainParams.y = state.y
                                windowManager.updateViewLayout(mainView, mainParams)

                                val capsuleLeft = (screenWidthPixels - closeCapsuleWidthPx) / 2f
                                val capsuleTop = (
                                    screenHeightPixels - closeCapsuleBottomMarginPx - closeCapsuleHeightPx
                                ).toFloat()
                                val isOverCloseTarget = isPointInsideCapsule(
                                    x = touchX,
                                    y = touchY,
                                    left = capsuleLeft,
                                    top = capsuleTop,
                                    width = closeCapsuleWidthPx.toFloat(),
                                    height = closeCapsuleHeightPx.toFloat(),
                                )
                                if (isOverCloseTarget && !closeTargetState.isActive) {
                                    closeTargetView?.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
                                }
                                closeTargetState.isActive = isOverCloseTarget
                            },
                            onDragEnded = {
                                val shouldClose = closeTargetState.isActive
                                closeTargetState.hide()
                                if (shouldClose) onCloseRequested()
                            },
                            onDragCancelled = closeTargetState::hide,
                            onResize = { dy ->
                                val oldHeight = state.heightPx
                                val newHeight = (oldHeight - dy.roundToInt())
                                    .coerceIn(state.minHeightPx, state.maxHeightPx)
                                val heightDiff = newHeight - oldHeight
                                state.heightPx = newHeight
                                state.y -= heightDiff
                                mainParams.y = state.y
                                mainParams.height = state.heightPx
                                windowManager.updateViewLayout(mainView, mainParams)
                            },
                            onToggleSize = {
                                val nextHeight = if (
                                    state.heightPx < (state.minHeightPx + state.maxHeightPx) / 2
                                ) {
                                    state.maxHeightPx
                                } else {
                                    state.minHeightPx
                                }
                                val heightDiff = nextHeight - state.heightPx
                                state.heightPx = nextHeight
                                state.y -= heightDiff
                                mainParams.y = state.y
                                mainParams.height = state.heightPx
                                windowManager.updateViewLayout(mainView, mainParams)
                            },
                        )
                    }
                }
            }

            val closeView = ComposeView(context).apply {
                setViewTreeLifecycleOwner(closeTargetOwner)
                setViewTreeViewModelStoreOwner(closeTargetOwner)
                setViewTreeSavedStateRegistryOwner(closeTargetOwner)
                setContent {
                    VoxlineTheme {
                        CloseTargetApp(closeTargetState)
                    }
                }
            }

            windowManager.addView(captionView, mainParams)
            windowManager.addView(closeView, closeTargetParams)

            mainView = captionView
            closeTargetView = closeView
            mainLifecycle = mainOwner
            closeTargetLifecycle = closeTargetOwner
        }
    }

    fun dismiss() {
        mainHandler.post {
            mainView?.let { windowManager.removeView(it) }
            closeTargetView?.let { windowManager.removeView(it) }
            mainView = null
            closeTargetView = null
            mainLifecycle?.destroy()
            closeTargetLifecycle?.destroy()
            mainLifecycle = null
            closeTargetLifecycle = null
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun FloatingCaptionApp(
    barHeightPx: Int,
    onDragStarted: () -> Unit,
    onMove: (Float, Float, Float, Float) -> Unit,
    onDragEnded: () -> Unit,
    onDragCancelled: () -> Unit,
    onResize: (Float) -> Unit,
    onToggleSize: () -> Unit,
) {
    val captionsState by VoxlineRuntimeStore.state.collectAsState()
    val barHeight = with(LocalDensity.current) { barHeightPx.toDp() }

    Surface(
        modifier = Modifier.fillMaxSize(),
        shape = RoundedCornerShape(20.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.94f),
        contentColor = MaterialTheme.colorScheme.onSurface,
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            ControlBarApp(
                modifier = Modifier.fillMaxWidth().height(barHeight),
                onDragStarted = onDragStarted,
                onMove = onMove,
                onDragEnded = onDragEnded,
                onDragCancelled = onDragCancelled,
                onResize = onResize,
                onToggleSize = onToggleSize,
            )
            VoxlineContentList(
                lines = captionsState.lines,
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .padding(horizontal = 16.dp, vertical = 8.dp),
            )
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun ControlBarApp(
    modifier: Modifier = Modifier,
    onDragStarted: () -> Unit,
    onMove: (Float, Float, Float, Float) -> Unit,
    onDragEnded: () -> Unit,
    onDragCancelled: () -> Unit,
    onResize: (Float) -> Unit,
    onToggleSize: () -> Unit,
) {
    var isHoveringHandle by remember { mutableStateOf(false) }
    val barAlpha by animateFloatAsState(
        targetValue = if (isHoveringHandle) 0.8f else 0.3f,
        animationSpec = MaterialTheme.motionScheme.fastEffectsSpec(),
        label = "Alpha",
    )
    val touchSlop = androidx.compose.ui.platform.LocalViewConfiguration.current.touchSlop
    val resizeHandleTouchWidthPx = with(LocalDensity.current) { 80.dp.toPx() }

    Box(modifier = modifier, contentAlignment = Alignment.Center) {
            val gestureState = remember {
                object {
                    var lastX = 0f
                    var lastY = 0f
                    var initialY = 0f
                    var isResizeGesture = false
                    var isDragging = false
                }
            }

            AndroidView(
                factory = { ctx -> View(ctx) },
                update = { view ->
                    view.setOnTouchListener { _, event ->
                        when (event.actionMasked) {
                            MotionEvent.ACTION_DOWN -> {
                                gestureState.lastX = event.rawX
                                gestureState.lastY = event.rawY
                                gestureState.initialY = event.rawY
                                gestureState.isResizeGesture =
                                    abs(event.x - view.width / 2f) <= resizeHandleTouchWidthPx / 2f
                                gestureState.isDragging = false
                                isHoveringHandle = gestureState.isResizeGesture
                                if (!gestureState.isResizeGesture) onDragStarted()
                                true
                            }

                            MotionEvent.ACTION_MOVE -> {
                                if (gestureState.isResizeGesture) {
                                    val currentY = event.rawY
                                    if (
                                        !gestureState.isDragging &&
                                        abs(currentY - gestureState.initialY) > touchSlop
                                    ) {
                                        gestureState.isDragging = true
                                    }
                                    if (gestureState.isDragging) {
                                        onResize(currentY - gestureState.lastY)
                                    }
                                    gestureState.lastY = currentY
                                } else {
                                    onMove(
                                        event.rawX - gestureState.lastX,
                                        event.rawY - gestureState.lastY,
                                        event.rawX,
                                        event.rawY,
                                    )
                                    gestureState.lastX = event.rawX
                                    gestureState.lastY = event.rawY
                                }
                                true
                            }

                            MotionEvent.ACTION_UP -> {
                                if (gestureState.isResizeGesture) {
                                    isHoveringHandle = false
                                    if (!gestureState.isDragging) onToggleSize()
                                } else {
                                    onDragEnded()
                                }
                                true
                            }

                            MotionEvent.ACTION_CANCEL -> {
                                if (gestureState.isResizeGesture) {
                                    isHoveringHandle = false
                                } else {
                                    onDragCancelled()
                                }
                                true
                            }

                            else -> true
                        }
                    }
                },
                modifier = Modifier.fillMaxSize(),
            )

            Box(
                modifier = Modifier
                    .size(36.dp, 5.dp)
                    .offset(y = 5.dp)
                    .zIndex(1f)
                    .clip(RoundedCornerShape(2.5.dp))
                    .background(Color.White.copy(alpha = barAlpha)),
            )
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun CloseTargetApp(state: CloseTargetState) {
    val motionScheme = MaterialTheme.motionScheme
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.BottomCenter) {
        AnimatedVisibility(
            visible = state.isVisible,
            enter = slideInVertically(
                animationSpec = motionScheme.defaultSpatialSpec(),
                initialOffsetY = { it },
            ) + fadeIn(animationSpec = motionScheme.defaultEffectsSpec()),
            exit = slideOutVertically(
                animationSpec = motionScheme.fastSpatialSpec(),
                targetOffsetY = { it },
            ) + fadeOut(animationSpec = motionScheme.fastEffectsSpec()),
        ) {
            Box(modifier = Modifier.padding(bottom = 20.dp)) {
                Button(
                    shapes = ButtonDefaults.shapes(),
                    onClick = {},
                    modifier = Modifier.width(128.dp).height(48.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (state.isActive) {
                            MaterialTheme.colorScheme.error
                        } else {
                            MaterialTheme.colorScheme.surfaceVariant
                        },
                        contentColor = if (state.isActive) {
                            MaterialTheme.colorScheme.onError
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        },
                    ),
                ) {
                    Text(text = t("stop"))
                }
            }
        }
    }
}

private fun isPointInsideCapsule(
    x: Float,
    y: Float,
    left: Float,
    top: Float,
    width: Float,
    height: Float,
): Boolean {
    val radius = height / 2f
    val centerY = top + radius
    val leftCircleCenterX = left + radius
    val rightCircleCenterX = left + width - radius

    return when {
        x in leftCircleCenterX..rightCircleCenterX && y in top..(top + height) -> true
        x < leftCircleCenterX -> (x - leftCircleCenterX) * (x - leftCircleCenterX) +
            (y - centerY) * (y - centerY) <= radius * radius
        x > rightCircleCenterX -> (x - rightCircleCenterX) * (x - rightCircleCenterX) +
            (y - centerY) * (y - centerY) <= radius * radius
        else -> false
    }
}

@Composable
fun VoxlineContentList(lines: List<VoxlineLine>, modifier: Modifier = Modifier) {
    val listState = rememberLazyListState()
    val isAtBottom by remember { derivedStateOf { listState.firstVisibleItemIndex <= 1 } }

    LaunchedEffect(lines.size) {
        if (lines.isNotEmpty() && isAtBottom) {
            listState.animateScrollToItem(0)
        }
    }

    LazyColumn(
        state = listState,
        reverseLayout = true,
        verticalArrangement = Arrangement.spacedBy(12.dp),
        modifier = modifier,
    ) {
        items(lines.reversed(), key = { it.id }) { line ->
            VoxlineLineItem(line = line, isNewest = line.id == lines.lastOrNull()?.id)
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
fun VoxlineLineItem(line: VoxlineLine, isNewest: Boolean) {
    Column(
        modifier = Modifier.animateContentSize(
            animationSpec = MaterialTheme.motionScheme.defaultSpatialSpec(),
        ),
    ) {
        val style = MaterialTheme.typography.titleLargeEmphasized
        if (isNewest && line.showTypewriter) {
            TypewriterText(text = line.sourceText, style = style)
        } else {
            Text(text = line.sourceText, style = style)
        }

        if (line.isTranslating && line.translatedText == null) {
            Text(
                text = "...",
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f),
                style = MaterialTheme.typography.bodyLarge,
            )
        } else if (!line.translatedText.isNullOrBlank()) {
            if (isNewest) {
                TypewriterText(
                    text = line.translatedText,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodyLarge,
                )
            } else {
                Text(
                    text = line.translatedText,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
        }
    }
}

@Composable
fun TypewriterText(
    text: String,
    modifier: Modifier = Modifier,
    color: Color = Color.Unspecified,
    style: androidx.compose.ui.text.TextStyle = androidx.compose.material3.LocalTextStyle.current,
    fontWeight: FontWeight? = null,
) {
    var displayedText by rememberSaveable { mutableStateOf("") }

    LaunchedEffect(text) {
        val keepLength = displayedText.length.coerceAtMost(text.length)
        if (displayedText != text.substring(0, keepLength)) {
            displayedText = text.substring(0, keepLength)
        }

        val charsToType = text.length - displayedText.length
        if (charsToType > 0) {
            val frameDelay = 16L
            val durationMs = (charsToType * 20L).coerceIn(150L, 800L)
            val charsPerFrame = (charsToType.toFloat() / (durationMs / frameDelay)).coerceAtLeast(1f)
            var currentLength = displayedText.length.toFloat()
            while (currentLength < text.length) {
                delay(frameDelay)
                currentLength += charsPerFrame
                val nextLength = currentLength.toInt().coerceAtMost(text.length)
                displayedText = text.substring(0, nextLength)
            }
        }
    }

    Text(
        text = displayedText,
        modifier = modifier,
        color = color,
        style = style,
        fontWeight = fontWeight,
    )
}
