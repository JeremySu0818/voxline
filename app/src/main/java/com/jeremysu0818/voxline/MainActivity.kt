package com.jeremysu0818.voxline

import android.Manifest
import android.accessibilityservice.AccessibilityServiceInfo
import android.app.Activity
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.view.accessibility.AccessibilityManager
import android.widget.Toast
import androidx.annotation.DrawableRes
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.ContentTransform
import androidx.compose.animation.EnterTransition
import androidx.compose.animation.ExitTransition
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExperimentalMaterial3ExpressiveApi
import androidx.compose.material3.ExposedDropdownMenuAnchorType
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearWavyProgressIndicator
import androidx.compose.material3.LoadingIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuDefaults
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.ProgressIndicatorDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import com.jeremysu0818.voxline.accessibility.VoxlineAccessibilityService
import com.jeremysu0818.voxline.data.VoxlineLanguage
import com.jeremysu0818.voxline.data.VoxlineLanguages
import com.jeremysu0818.voxline.data.VoxlineRuntimeState
import com.jeremysu0818.voxline.data.VoxlineSettings
import com.jeremysu0818.voxline.data.I18n
import com.jeremysu0818.voxline.data.SpeechEngineOption
import com.jeremysu0818.voxline.data.ThemeMode
import com.jeremysu0818.voxline.data.WhisperModelOption
import com.jeremysu0818.voxline.data.t
import com.jeremysu0818.voxline.service.VoxlineCaptureService
import com.jeremysu0818.voxline.ui.theme.VoxlineTheme
import com.jeremysu0818.voxline.whisper.ModelDownloadState
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private var startRequestCount by mutableIntStateOf(0)
    private var resumeCount by mutableIntStateOf(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        VoxlineGraph.ensureInitialized(this)
        enableEdgeToEdge()
        if (intent?.action == ACTION_START_FROM_TILE) {
            startRequestCount++
        }
        setContent {
            val settings by VoxlineGraph.preferences.settings.collectAsState()
            VoxlineTheme(themeMode = settings.themeMode) {
                VoxlineApp(
                    startRequestCount = startRequestCount,
                    resumeCount = resumeCount,
                    onStartRequested = { startRequestCount++ },
                )
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (intent.action == ACTION_START_FROM_TILE) {
            startRequestCount++
        }
    }

    override fun onResume() {
        super.onResume()
        resumeCount++
    }

    companion object {
        const val ACTION_START_FROM_TILE = "com.jeremysu0818.voxline.action.START_FROM_TILE"
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun VoxlineApp(
    startRequestCount: Int,
    resumeCount: Int,
    onStartRequested: () -> Unit,
) {
    val context = LocalContext.current
    val settings by VoxlineGraph.preferences.settings.collectAsState()
    val runtimeState by VoxlineGraph.runtimeStore.state.collectAsState()
    val downloadStates by VoxlineGraph.modelRepository.downloadStates.collectAsState()
    val scope = rememberCoroutineScope()
    var selectedDestinationIndex by rememberSaveable { mutableIntStateOf(0) }

    var permissionRefresh by remember { mutableIntStateOf(0) }
    var pendingStart by remember { mutableStateOf(false) }
    var overlayPrompted by remember { mutableStateOf(false) }
    var recordPrompted by remember { mutableStateOf(false) }
    var notificationPrompted by remember { mutableStateOf(false) }
    var accessibilityPrompted by remember { mutableStateOf(false) }
    var isMlKitAdvancedAvailable by remember { mutableStateOf<Boolean?>(null) }
    val downloadJobs = remember { mutableMapOf<WhisperModelOption, Job>() }

    val mediaProjectionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            VoxlineCaptureService.start(context, result.resultCode, result.data!!)
        } else {
            VoxlineGraph.runtimeStore.setStopped(I18n.getString("error_projection_cancelled"))
        }
    }

    val overlaySettingsLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) {
        permissionRefresh++
    }

    val accessibilitySettingsLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartActivityForResult(),
    ) {
        permissionRefresh++
    }

    val recordPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (!granted) {
            pendingStart = false
            VoxlineGraph.runtimeStore.setError(I18n.getString("error_no_record"))
        }
        permissionRefresh++
    }

    val notificationPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (!granted) {
            pendingStart = false
            VoxlineGraph.runtimeStore.setError(I18n.getString("error_no_notification"))
        }
        permissionRefresh++
    }

    val overlayGranted = remember(resumeCount, permissionRefresh) {
        Settings.canDrawOverlays(context)
    }
    val recordGranted = remember(resumeCount, permissionRefresh) {
        context.hasPermission(Manifest.permission.RECORD_AUDIO)
    }
    val notificationGranted = remember(resumeCount, permissionRefresh) {
        Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
            context.hasPermission(Manifest.permission.POST_NOTIFICATIONS)
    }
    val accessibilityGranted = remember(resumeCount, permissionRefresh) {
        context.isVoxlineAccessibilityServiceEnabled()
    }
    val accessibilityConnected by VoxlineAccessibilityService.isConnected.collectAsState()

    LaunchedEffect(settings.model) {
        VoxlineGraph.modelRepository.refresh(settings.model)
    }

    LaunchedEffect(settings.sourceLanguageTag, settings.translationEnabled) {
        val advancedSourceTag = VoxlineLanguages.compatibleSourceTag(
            tag = settings.sourceLanguageTag,
            engine = SpeechEngineOption.MLKIT_ADVANCED,
            translationEnabled = settings.translationEnabled,
        )
        isMlKitAdvancedAvailable = advancedSourceTag != null && runCatching {
            VoxlineGraph.mlKitSpeechTranscriber.isAdvancedAvailable(advancedSourceTag)
        }.getOrDefault(false)
    }

    LaunchedEffect(isMlKitAdvancedAvailable, settings.speechEngine) {
        if (
            isMlKitAdvancedAvailable == false &&
            settings.speechEngine == SpeechEngineOption.MLKIT_ADVANCED
        ) {
            VoxlineGraph.preferences.updateSpeechEngine(SpeechEngineOption.MLKIT_BASIC)
        }
    }

    LaunchedEffect(startRequestCount) {
        if (startRequestCount > 0) {
            pendingStart = true
            overlayPrompted = false
            recordPrompted = false
            notificationPrompted = false
            accessibilityPrompted = false
        }
    }

    LaunchedEffect(
        pendingStart,
        overlayGranted,
        recordGranted,
        notificationGranted,
        accessibilityGranted,
        accessibilityConnected,
        permissionRefresh,
        resumeCount,
    ) {
        if (!pendingStart) return@LaunchedEffect
        when {
            !accessibilityGranted -> {
                if (!accessibilityPrompted) {
                    accessibilityPrompted = true
                    accessibilitySettingsLauncher.launch(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                }
            }

            !accessibilityConnected -> {
                pendingStart = false
                VoxlineGraph.runtimeStore.setError(I18n.getString("error_accessibility_service_unavailable"))
            }

            !overlayGranted -> {
                if (!overlayPrompted) {
                    overlayPrompted = true
                    overlaySettingsLauncher.launch(context.overlaySettingsIntent())
                }
            }

            !recordGranted -> {
                if (!recordPrompted) {
                    recordPrompted = true
                    recordPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                }
            }

            !notificationGranted -> {
                if (!notificationPrompted && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    notificationPrompted = true
                    notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                }
            }

            else -> {
                pendingStart = false
                val projectionManager = context.getSystemService(MediaProjectionManager::class.java)
                mediaProjectionLauncher.launch(projectionManager.createScreenCaptureIntent())
            }
        }
    }

    val allPermissionsGranted =
        accessibilityGranted && overlayGranted && recordGranted && notificationGranted

    val selectedDestination = AppDestination.entries[selectedDestinationIndex]
    val pageTransition = expressiveFadeTransform()

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        text = t(selectedDestination.titleKey),
                        style = MaterialTheme.typography.titleLargeEmphasized,
                    )
                },
            )
        },
        bottomBar = {
            ExpressiveNavigationBar(
                selectedDestination = selectedDestination,
                onDestinationSelected = { selectedDestinationIndex = it.ordinal },
            )
        },
    ) { innerPadding ->
        AnimatedContent(
            targetState = selectedDestination,
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding),
            transitionSpec = { pageTransition },
            label = "app_destination",
        ) { destination ->
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(
                    start = 20.dp,
                    end = 20.dp,
                    top = 12.dp,
                    bottom = 32.dp,
                ),
            ) {
                item {
                    Box(
                        modifier = Modifier.fillMaxWidth(),
                        contentAlignment = Alignment.TopCenter,
                    ) {
                        Column(
                            modifier = Modifier
                                .widthIn(max = 720.dp)
                                .fillMaxWidth(),
                            verticalArrangement = Arrangement.spacedBy(
                                if (destination == AppDestination.Home) 20.dp else 12.dp,
                            ),
                        ) {
                            when (destination) {
                                AppDestination.Home -> {
                                    ControlCenterCard(
                                        runtimeState = runtimeState,
                                        isRunning = runtimeState.isRunning,
                                        canStart = accessibilityGranted && accessibilityConnected &&
                                            overlayGranted && recordGranted,
                                        onStart = onStartRequested,
                                        onStop = { VoxlineCaptureService.stop(context) },
                                    )

                                    AnimatedVisibility(
                                        visible = !allPermissionsGranted,
                                        enter = expressiveEnter(),
                                        exit = expressiveExit(),
                                    ) {
                                        PermissionAlertCard(
                                            overlayGranted = overlayGranted,
                                            recordGranted = recordGranted,
                                            notificationGranted = notificationGranted,
                                            accessibilityGranted = accessibilityGranted,
                                            onOpenAccessibilitySettings = {
                                                accessibilityPrompted = true
                                                accessibilitySettingsLauncher.launch(
                                                    Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS),
                                                )
                                            },
                                            onOpenOverlaySettings = {
                                                overlayPrompted = true
                                                overlaySettingsLauncher.launch(context.overlaySettingsIntent())
                                            },
                                            onRequestRecord = {
                                                recordPrompted = true
                                                recordPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                                            },
                                            onRequestNotifications = {
                                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                                    notificationPrompted = true
                                                    notificationPermissionLauncher.launch(
                                                        Manifest.permission.POST_NOTIFICATIONS,
                                                    )
                                                }
                                            },
                                        )
                                    }

                                    QuickSettingsCard(
                                        settings = settings,
                                        isMlKitAdvancedAvailable = isMlKitAdvancedAvailable != false,
                                        onTranslationEnabledChanged =
                                            VoxlineGraph.preferences::updateTranslationEnabled,
                                        onEngineSelected = VoxlineGraph.preferences::updateSpeechEngine,
                                    )
                                }

                                AppDestination.Settings -> {
                                    SettingsCard(
                                        icon = R.drawable.sym_settings,
                                        title = t("speech_engine"),
                                    ) {
                                        SpeechEngineSection(
                                            settings = settings,
                                            isMlKitAdvancedAvailable = isMlKitAdvancedAvailable != false,
                                            onEngineSelected = VoxlineGraph.preferences::updateSpeechEngine,
                                            onUnsupportedAdvancedSelected = {
                                                Toast.makeText(
                                                    context,
                                                    I18n.getString("mlkit_advanced_unsupported"),
                                                    Toast.LENGTH_SHORT,
                                                ).show()
                                            },
                                        )
                                    }

                                    AnimatedVisibility(
                                        visible = settings.speechEngine == SpeechEngineOption.WHISPER,
                                        enter = expressiveEnter(),
                                        exit = expressiveExit(),
                                    ) {
                                        SettingsCard(
                                            icon = R.drawable.sym_download,
                                            title = t("whisper_model"),
                                        ) {
                                            WhisperModelSection(
                                                settings = settings,
                                                downloadStates = downloadStates,
                                                onModelSelected = VoxlineGraph.preferences::updateModel,
                                                onDownloadModel = { model ->
                                                    if (downloadJobs[model]?.isActive == true) {
                                                        return@WhisperModelSection
                                                    }
                                                    val job = scope.launch(start = CoroutineStart.LAZY) {
                                                        try {
                                                            VoxlineGraph.modelRepository.ensureModel(model)
                                                        } finally {
                                                            downloadJobs.remove(model)
                                                        }
                                                    }
                                                    downloadJobs[model] = job
                                                    job.start()
                                                },
                                                onCancelDownload = { model ->
                                                    downloadJobs.remove(model)?.cancel()
                                                },
                                                onDeleteModel = { model ->
                                                    scope.launch {
                                                        VoxlineGraph.modelRepository.deleteModel(model)
                                                    }
                                                },
                                            )
                                        }
                                    }

                                    SettingsCard(
                                        icon = R.drawable.sym_translate,
                                        title = t("local_translation"),
                                        trailingContent = {
                                            Switch(
                                                checked = settings.translationEnabled,
                                                onCheckedChange = VoxlineGraph.preferences::updateTranslationEnabled,
                                            )
                                        },
                                    ) {
                                        TranslationSection(
                                            settings = settings,
                                            onSourceChanged = VoxlineGraph.preferences::updateSourceLanguage,
                                            onTargetChanged = VoxlineGraph.preferences::updateTargetLanguage,
                                        )
                                    }

                                    SettingsCard(
                                        icon = R.drawable.sym_language,
                                        title = t("ui_language"),
                                    ) {
                                        UiLanguageSection(
                                            settings = settings,
                                            onLanguageSelected = VoxlineGraph.preferences::updateUiLanguage,
                                        )
                                    }

                                    SettingsCard(
                                        icon = R.drawable.sym_palette,
                                        title = t("color_theme"),
                                    ) {
                                        ThemeModeSection(
                                            settings = settings,
                                            onThemeModeSelected = VoxlineGraph.preferences::updateThemeMode,
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

private enum class AppDestination(
    val titleKey: String,
    @param:DrawableRes val icon: Int,
) {
    Home("app_name", R.drawable.sym_home),
    Settings("settings", R.drawable.sym_settings),
}

@Composable
private fun ExpressiveNavigationBar(
    selectedDestination: AppDestination,
    onDestinationSelected: (AppDestination) -> Unit,
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .navigationBarsPadding()
            .padding(horizontal = 20.dp, vertical = 10.dp),
        contentAlignment = Alignment.Center,
    ) {
        NavigationBar(
            modifier = Modifier
                .widthIn(max = 720.dp)
                .fillMaxWidth()
                .clip(MaterialTheme.shapes.extraLargeIncreased),
            containerColor = MaterialTheme.colorScheme.surfaceContainerHigh,
            tonalElevation = 6.dp,
            windowInsets = WindowInsets(0, 0, 0, 0),
        ) {
            AppDestination.entries.forEach { destination ->
                NavigationBarItem(
                    selected = destination == selectedDestination,
                    onClick = { onDestinationSelected(destination) },
                    icon = {
                        Icon(
                            painter = painterResource(destination.icon),
                            contentDescription = null,
                        )
                    },
                    label = { Text(t(destination.titleKey)) },
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun QuickSettingsCard(
    settings: VoxlineSettings,
    isMlKitAdvancedAvailable: Boolean,
    onTranslationEnabledChanged: (Boolean) -> Unit,
    onEngineSelected: (SpeechEngineOption) -> Unit,
) {
    val colorScheme = MaterialTheme.colorScheme
    val motionScheme = MaterialTheme.motionScheme
    val engines = remember(isMlKitAdvancedAvailable) {
        SpeechEngineOption.entries
            .filter { option ->
                option != SpeechEngineOption.MLKIT_ADVANCED || isMlKitAdvancedAvailable
            }
            .map { option -> VoxlineLanguage(tag = option.id, label = option.label) }
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.largeIncreased,
        colors = CardDefaults.cardColors(containerColor = colorScheme.surfaceContainerLow),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .animateContentSize(animationSpec = motionScheme.defaultSpatialSpec()),
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 20.dp, top = 16.dp, end = 16.dp, bottom = 16.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Surface(
                    modifier = Modifier.size(40.dp),
                    shape = CircleShape,
                    color = colorScheme.secondaryContainer,
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            painter = painterResource(R.drawable.sym_translate),
                            contentDescription = null,
                            tint = colorScheme.onSecondaryContainer,
                        )
                    }
                }
                Text(
                    text = t("local_translation"),
                    style = MaterialTheme.typography.titleMediumEmphasized,
                    modifier = Modifier.weight(1f),
                )
                Switch(
                    checked = settings.translationEnabled,
                    onCheckedChange = onTranslationEnabledChanged,
                )
            }

            HorizontalDivider(color = colorScheme.outlineVariant.copy(alpha = 0.45f))

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = t("speech_engine"),
                    style = MaterialTheme.typography.labelLarge,
                    color = colorScheme.onSurfaceVariant,
                )
                LanguageDropdown(
                    selectedTag = settings.speechEngine.id,
                    selectedLabel = settings.speechEngine.label,
                    languages = engines,
                    onSelected = { selectedId ->
                        onEngineSelected(SpeechEngineOption.fromId(selectedId))
                    },
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun expressiveEnter(): EnterTransition {
    val motionScheme = MaterialTheme.motionScheme
    return expandVertically(animationSpec = motionScheme.defaultSpatialSpec()) +
        fadeIn(animationSpec = motionScheme.defaultEffectsSpec())
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun expressiveExit(): ExitTransition {
    val motionScheme = MaterialTheme.motionScheme
    return shrinkVertically(animationSpec = motionScheme.fastSpatialSpec()) +
        fadeOut(animationSpec = motionScheme.fastEffectsSpec())
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun expressiveFadeTransform(): ContentTransform {
    val motionScheme = MaterialTheme.motionScheme
    return fadeIn(animationSpec = motionScheme.defaultEffectsSpec())
        .togetherWith(fadeOut(animationSpec = motionScheme.fastEffectsSpec()))
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun expressiveContentTransform(): ContentTransform {
    val motionScheme = MaterialTheme.motionScheme
    return (
        fadeIn(animationSpec = motionScheme.defaultEffectsSpec()) +
            scaleIn(initialScale = 0.96f, animationSpec = motionScheme.defaultSpatialSpec())
        ).togetherWith(fadeOut(animationSpec = motionScheme.fastEffectsSpec()))
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun ControlCenterCard(
    runtimeState: VoxlineRuntimeState,
    isRunning: Boolean,
    canStart: Boolean,
    onStart: () -> Unit,
    onStop: () -> Unit,
) {
    val colorScheme = MaterialTheme.colorScheme
    val motionScheme = MaterialTheme.motionScheme
    val contentTransform = expressiveContentTransform()
    val hasError = runtimeState.errorMessage != null
    val containerColor by animateColorAsState(
        targetValue = when {
            hasError -> colorScheme.errorContainer
            isRunning -> colorScheme.primaryContainer
            else -> colorScheme.surfaceContainerHigh
        },
        animationSpec = motionScheme.defaultEffectsSpec(),
        label = "caption_control_container",
    )
    val iconColor = when {
        hasError -> colorScheme.onErrorContainer
        isRunning -> colorScheme.onPrimary
        else -> colorScheme.primary
    }
    val iconContainerColor = when {
        hasError -> colorScheme.error
        isRunning -> colorScheme.primary
        else -> colorScheme.primaryContainer
    }
    val statusIcon = when {
        hasError -> R.drawable.sym_error
        isRunning -> R.drawable.sym_check_circle
        else -> R.drawable.sym_mic
    }

    ElevatedCard(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.extraLargeIncreased,
        colors = CardDefaults.elevatedCardColors(containerColor = containerColor),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp)
                .animateContentSize(animationSpec = motionScheme.defaultSpatialSpec()),
            verticalArrangement = Arrangement.spacedBy(20.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Surface(
                    modifier = Modifier.size(52.dp),
                    shape = CircleShape,
                    color = iconContainerColor,
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            painter = painterResource(statusIcon),
                            contentDescription = null,
                            tint = iconColor,
                        )
                    }
                }
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = t("status"),
                        style = MaterialTheme.typography.labelLarge,
                        color = if (hasError) colorScheme.onErrorContainer else colorScheme.onSurfaceVariant,
                    )
                    AnimatedContent(
                        targetState = runtimeState.status,
                        label = "caption_status",
                        transitionSpec = {
                            val downloadingPrefix = I18n.getString("model_downloading_status")
                                .substringBefore("{")
                                .trim()
                            val isDownloading = downloadingPrefix.isNotEmpty() &&
                                (targetState.startsWith(downloadingPrefix) ||
                                    initialState.startsWith(downloadingPrefix))
                            if (isDownloading) {
                                ContentTransform(
                                    targetContentEnter = EnterTransition.None,
                                    initialContentExit = ExitTransition.None,
                                )
                            } else {
                                contentTransform
                            }
                        },
                    ) { targetStatus ->
                        Text(
                            text = t(targetStatus),
                            style = MaterialTheme.typography.titleLargeEmphasized,
                        )
                    }
                }
            }

            val lastLine = runtimeState.lines.lastOrNull()
            AnimatedVisibility(
                visible = lastLine != null && lastLine.sourceText.isNotBlank(),
                enter = expressiveEnter(),
                exit = expressiveExit(),
            ) {
                Surface(
                    shape = MaterialTheme.shapes.medium,
                    color = if (hasError) {
                        colorScheme.errorContainer.copy(alpha = 0.48f)
                    } else {
                        colorScheme.surfaceContainerLowest
                    },
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                    ) {
                        Text(
                            text = lastLine?.sourceText.orEmpty(),
                            style = MaterialTheme.typography.bodyLarge,
                        )
                        AnimatedVisibility(
                            visible = lastLine?.isTranslating == true &&
                                lastLine.translatedText.isNullOrBlank(),
                            enter = expressiveEnter(),
                            exit = expressiveExit(),
                        ) {
                            Text(
                                text = t("translating"),
                                style = MaterialTheme.typography.bodyMedium,
                                color = colorScheme.onSurfaceVariant,
                            )
                        }
                        AnimatedVisibility(
                            visible = !lastLine?.translatedText.isNullOrBlank(),
                            enter = expressiveEnter(),
                            exit = expressiveExit(),
                        ) {
                            Text(
                                text = lastLine?.translatedText.orEmpty(),
                                style = MaterialTheme.typography.bodyMedium,
                                color = colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }

            AnimatedVisibility(
                visible = runtimeState.errorMessage != null,
                enter = fadeIn(animationSpec = motionScheme.defaultEffectsSpec()),
                exit = fadeOut(animationSpec = motionScheme.fastEffectsSpec()),
            ) {
                Text(
                    text = t(runtimeState.errorMessage.orEmpty()),
                    style = MaterialTheme.typography.bodyMedium,
                    color = colorScheme.onErrorContainer,
                )
            }

            AnimatedContent(
                targetState = isRunning,
                label = "caption_primary_action",
                transitionSpec = { contentTransform },
            ) { running ->
                if (running) {
                    OutlinedButton(
                        shapes = ButtonDefaults.shapes(),
                        modifier = Modifier.fillMaxWidth(),
                        onClick = onStop,
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = colorScheme.error,
                        ),
                        border = BorderStroke(1.dp, colorScheme.error),
                    ) {
                        Icon(
                            painter = painterResource(R.drawable.sym_stop_circle),
                            contentDescription = null,
                        )
                        Spacer(modifier = Modifier.size(ButtonDefaults.IconSpacing))
                        Text(t("stop"))
                    }
                } else {
                    Button(
                        shapes = ButtonDefaults.shapes(),
                        modifier = Modifier.fillMaxWidth(),
                        onClick = onStart,
                    ) {
                        Icon(
                            painter = painterResource(R.drawable.sym_mic),
                            contentDescription = null,
                        )
                        Spacer(modifier = Modifier.size(ButtonDefaults.IconSpacing))
                        Text(if (canStart) t("start_caption") else t("check_permission"))
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun PermissionAlertCard(
    overlayGranted: Boolean,
    recordGranted: Boolean,
    notificationGranted: Boolean,
    accessibilityGranted: Boolean,
    onOpenOverlaySettings: () -> Unit,
    onOpenAccessibilitySettings: () -> Unit,
    onRequestRecord: () -> Unit,
    onRequestNotifications: () -> Unit,
) {
    val colorScheme = MaterialTheme.colorScheme
    val motionScheme = MaterialTheme.motionScheme
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.largeIncreased,
        colors = CardDefaults.cardColors(
            containerColor = colorScheme.errorContainer,
            contentColor = colorScheme.onErrorContainer,
        ),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp)
                .animateContentSize(animationSpec = motionScheme.defaultSpatialSpec()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Surface(
                    modifier = Modifier.size(40.dp),
                    shape = CircleShape,
                    color = colorScheme.error,
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            painter = painterResource(R.drawable.sym_warning),
                            contentDescription = null,
                            tint = colorScheme.onError,
                        )
                    }
                }
                Text(
                    text = t("permission_required"),
                    style = MaterialTheme.typography.titleMediumEmphasized,
                )
            }
            Text(
                text = t("permission_reason"),
                style = MaterialTheme.typography.bodyMedium,
            )

            if (!accessibilityGranted) {
                PermissionRow(
                    icon = R.drawable.sym_accessibility_new,
                    label = t("permission_accessibility"),
                    actionText = t("open_settings"),
                    onAction = onOpenAccessibilitySettings,
                )
            }
            if (!overlayGranted) {
                PermissionRow(
                    icon = R.drawable.sym_settings,
                    label = t("permission_overlay"),
                    actionText = t("open_settings"),
                    onAction = onOpenOverlaySettings,
                )
            }
            if (!recordGranted) {
                PermissionRow(
                    icon = R.drawable.sym_mic,
                    label = t("permission_record"),
                    actionText = t("allow"),
                    onAction = onRequestRecord,
                )
            }
            if (!notificationGranted && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                PermissionRow(
                    icon = R.drawable.sym_notifications,
                    label = t("permission_notification"),
                    actionText = t("allow"),
                    onAction = onRequestNotifications,
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun PermissionRow(
    @DrawableRes icon: Int,
    label: String,
    actionText: String,
    onAction: () -> Unit,
) {
    val colorScheme = MaterialTheme.colorScheme
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Surface(
            modifier = Modifier.size(36.dp),
            shape = CircleShape,
            color = colorScheme.onErrorContainer.copy(alpha = 0.1f),
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(
                    painter = painterResource(icon),
                    contentDescription = null,
                    tint = colorScheme.onErrorContainer,
                )
            }
        }
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            modifier = Modifier.weight(1f),
        )
        TextButton(
            shapes = ButtonDefaults.shapes(),
            onClick = onAction,
            colors = ButtonDefaults.textButtonColors(contentColor = colorScheme.onErrorContainer),
        ) {
            Text(actionText)
        }
    }
}

@Composable
private fun SettingsHeading() {
    Row(
        modifier = Modifier.padding(start = 4.dp, top = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Surface(
            modifier = Modifier.size(36.dp),
            shape = CircleShape,
            color = MaterialTheme.colorScheme.secondaryContainer,
        ) {
            Box(contentAlignment = Alignment.Center) {
                Icon(
                    painter = painterResource(R.drawable.sym_settings),
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSecondaryContainer,
                )
            }
        }
        Text(
            text = t("settings"),
            style = MaterialTheme.typography.titleLargeEmphasized,
        )
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun SettingsCard(
    @DrawableRes icon: Int,
    title: String,
    trailingContent: @Composable (() -> Unit)? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    val colorScheme = MaterialTheme.colorScheme
    val motionScheme = MaterialTheme.motionScheme
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.largeIncreased,
        colors = CardDefaults.cardColors(containerColor = colorScheme.surfaceContainerLow),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .animateContentSize(animationSpec = motionScheme.defaultSpatialSpec()),
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 20.dp, top = 18.dp, end = 16.dp, bottom = 18.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Surface(
                    modifier = Modifier.size(40.dp),
                    shape = CircleShape,
                    color = colorScheme.primaryContainer,
                ) {
                    Box(contentAlignment = Alignment.Center) {
                        Icon(
                            painter = painterResource(icon),
                            contentDescription = null,
                            tint = colorScheme.onPrimaryContainer,
                        )
                    }
                }
                Text(
                    text = title,
                    style = MaterialTheme.typography.titleMediumEmphasized,
                    modifier = Modifier.weight(1f),
                )
                trailingContent?.invoke()
            }
            HorizontalDivider(
                color = colorScheme.outlineVariant.copy(alpha = 0.45f),
            )
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(20.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
                content = content,
            )
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun SpeechEngineSection(
    settings: VoxlineSettings,
    isMlKitAdvancedAvailable: Boolean,
    onEngineSelected: (SpeechEngineOption) -> Unit,
    onUnsupportedAdvancedSelected: () -> Unit,
) {
    val contentTransform = expressiveContentTransform()
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        SpeechEngineOption.entries.chunked(2).forEach { rowOptions ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                rowOptions.forEach { option ->
                    val isAdvancedOption = option == SpeechEngineOption.MLKIT_ADVANCED
                    val isOptionEnabled = !isAdvancedOption || isMlKitAdvancedAvailable
                    val isSelected = settings.speechEngine == option
                    FilterChip(
                        selected = isSelected,
                        onClick = {
                            if (isOptionEnabled) {
                                onEngineSelected(option)
                            } else {
                                onUnsupportedAdvancedSelected()
                            }
                        },
                        label = {
                            Text(
                                text = option.label,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis,
                            )
                        },
                        shapes = FilterChipDefaults.shapes(),
                        modifier = Modifier.weight(1f),
                        enabled = isOptionEnabled,
                        leadingIcon = if (isSelected) {
                            {
                                Icon(
                                    painter = painterResource(R.drawable.sym_check_circle),
                                    contentDescription = null,
                                    modifier = Modifier.size(FilterChipDefaults.IconSize),
                                )
                            }
                        } else {
                            null
                        },
                    )
                }
                if (rowOptions.size == 1) {
                    Spacer(modifier = Modifier.weight(1f))
                }
            }
        }
        AnimatedContent(
            targetState = settings.speechEngine,
            label = "engine_description",
            transitionSpec = { contentTransform },
        ) { engine ->
            Surface(
                shape = MaterialTheme.shapes.medium,
                color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.55f),
            ) {
                Text(
                    text = when (engine) {
                        SpeechEngineOption.WHISPER -> t("engine_whisper_desc")
                        SpeechEngineOption.MLKIT_BASIC -> t("engine_mlkit_basic_desc")
                        SpeechEngineOption.MLKIT_ADVANCED -> t("engine_mlkit_advanced_desc")
                    },
                    modifier = Modifier.padding(14.dp),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun WhisperModelSection(
    settings: VoxlineSettings,
    downloadStates: Map<WhisperModelOption, ModelDownloadState>,
    onModelSelected: (WhisperModelOption) -> Unit,
    onDownloadModel: (WhisperModelOption) -> Unit,
    onCancelDownload: (WhisperModelOption) -> Unit,
    onDeleteModel: (WhisperModelOption) -> Unit,
) {
    var showDeleteConfirmation by remember { mutableStateOf(false) }
    var showCancelConfirmation by remember { mutableStateOf(false) }

    if (showCancelConfirmation) {
        AlertDialog(
            onDismissRequest = { showCancelConfirmation = false },
            title = { Text(text = t("cancel_download_title")) },
            text = { Text(text = t("cancel_download_message", settings.model.displayName)) },
            confirmButton = {
                TextButton(
                    shapes = ButtonDefaults.shapes(),
                    onClick = {
                        showCancelConfirmation = false
                        onCancelDownload(settings.model)
                    },
                    colors = ButtonDefaults.textButtonColors(
                        contentColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Text(t("confirm_cancel"))
                }
            },
            dismissButton = {
                TextButton(
                    shapes = ButtonDefaults.shapes(),
                    onClick = { showCancelConfirmation = false },
                ) {
                    Text(t("back"))
                }
            },
        )
    }

    if (showDeleteConfirmation) {
        AlertDialog(
            onDismissRequest = { showDeleteConfirmation = false },
            title = { Text(text = t("delete_model_title")) },
            text = { Text(text = t("delete_model_message", settings.model.displayName)) },
            confirmButton = {
                TextButton(
                    shapes = ButtonDefaults.shapes(),
                    onClick = {
                        showDeleteConfirmation = false
                        onDeleteModel(settings.model)
                    },
                    colors = ButtonDefaults.textButtonColors(
                        contentColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Text(t("confirm_delete"))
                }
            },
            dismissButton = {
                TextButton(
                    shapes = ButtonDefaults.shapes(),
                    onClick = { showDeleteConfirmation = false },
                ) {
                    Text(t("cancel"))
                }
            },
        )
    }

    val selectedDownloadState = downloadStates[settings.model] ?: run {
        ModelDownloadState(model = settings.model)
    }
    val animatedDownloadProgress by animateFloatAsState(
        targetValue = selectedDownloadState.progress.coerceIn(0f, 1f),
        animationSpec = ProgressIndicatorDefaults.ProgressAnimationSpec,
        label = "model_download_progress",
    )
    val contentTransform = expressiveContentTransform()

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        WhisperModelOption.entries.chunked(2).forEach { rowModels ->
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                rowModels.forEach { option ->
                    val isSelected = settings.model == option
                    FilterChip(
                        selected = isSelected,
                        onClick = { onModelSelected(option) },
                        label = {
                            Text(
                                text = "${option.displayName} · ${option.sizeLabel}",
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis,
                            )
                        },
                        shapes = FilterChipDefaults.shapes(),
                        modifier = Modifier.weight(1f),
                        leadingIcon = if (isSelected) {
                            {
                                Icon(
                                    painter = painterResource(R.drawable.sym_check_circle),
                                    contentDescription = null,
                                    modifier = Modifier.size(FilterChipDefaults.IconSize),
                                )
                            }
                        } else {
                            null
                        },
                    )
                }
                if (rowModels.size == 1) {
                    Spacer(modifier = Modifier.weight(1f))
                }
            }
        }

        val hasError = selectedDownloadState.errorMessage != null
        val statusContainerColor = when {
            hasError -> MaterialTheme.colorScheme.errorContainer
            selectedDownloadState.isDownloaded -> MaterialTheme.colorScheme.secondaryContainer
            else -> MaterialTheme.colorScheme.surfaceContainerHighest
        }
        val statusContentColor = when {
            hasError -> MaterialTheme.colorScheme.onErrorContainer
            selectedDownloadState.isDownloaded -> MaterialTheme.colorScheme.onSecondaryContainer
            else -> MaterialTheme.colorScheme.onSurfaceVariant
        }
        val statusIcon = when {
            hasError -> R.drawable.sym_error
            selectedDownloadState.isDownloaded -> R.drawable.sym_check_circle
            else -> R.drawable.sym_download
        }
        Surface(
            shape = MaterialTheme.shapes.medium,
            color = statusContainerColor,
            contentColor = statusContentColor,
        ) {
            Row(
                modifier = Modifier.padding(14.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                if (selectedDownloadState.isDownloading) {
                    LoadingIndicator(
                        modifier = Modifier.size(28.dp),
                        color = MaterialTheme.colorScheme.primary,
                    )
                } else {
                    Icon(painter = painterResource(statusIcon), contentDescription = null)
                }
                Text(
                    text = when {
                        selectedDownloadState.isDownloaded -> t("model_downloaded")
                        selectedDownloadState.isDownloading -> selectedDownloadState.buildStatusText()
                        selectedDownloadState.errorMessage != null -> t(selectedDownloadState.errorMessage)
                        else -> t("model_not_downloaded")
                    },
                    modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }

        AnimatedVisibility(
            visible = selectedDownloadState.isDownloading,
            enter = expressiveEnter(),
            exit = expressiveExit(),
        ) {
            LinearWavyProgressIndicator(
                progress = { animatedDownloadProgress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(12.dp),
            )
        }

        AnimatedContent(
            targetState = when {
                selectedDownloadState.isDownloading -> ModelAction.Cancel
                selectedDownloadState.isDownloaded -> ModelAction.Delete
                else -> ModelAction.Download
            },
            label = "model_action",
            transitionSpec = { contentTransform },
        ) { action ->
            when (action) {
                ModelAction.Download -> {
                    FilledTonalButton(
                        shapes = ButtonDefaults.shapes(),
                        modifier = Modifier.fillMaxWidth(),
                        onClick = { onDownloadModel(settings.model) },
                        enabled = !selectedDownloadState.isDownloading && !selectedDownloadState.isDownloaded,
                    ) {
                        Icon(painter = painterResource(R.drawable.sym_download), contentDescription = null)
                        Spacer(modifier = Modifier.size(ButtonDefaults.IconSpacing))
                        Text(t("download_model", settings.model.displayName))
                    }
                }

                ModelAction.Cancel -> {
                    OutlinedButton(
                        shapes = ButtonDefaults.shapes(),
                        modifier = Modifier.fillMaxWidth(),
                        onClick = { showCancelConfirmation = true },
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = MaterialTheme.colorScheme.error,
                        ),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.error),
                    ) {
                        Text(t("cancel_download"))
                    }
                }

                ModelAction.Delete -> {
                    OutlinedButton(
                        shapes = ButtonDefaults.shapes(),
                        modifier = Modifier.fillMaxWidth(),
                        onClick = { showDeleteConfirmation = true },
                        colors = ButtonDefaults.outlinedButtonColors(
                            contentColor = MaterialTheme.colorScheme.error,
                        ),
                        border = BorderStroke(1.dp, MaterialTheme.colorScheme.error),
                    ) {
                        Text(t("delete_model", settings.model.displayName))
                    }
                }
            }
        }
    }
}

private enum class ModelAction {
    Download,
    Cancel,
    Delete,
}

@OptIn(ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun TranslationSection(
    settings: VoxlineSettings,
    onSourceChanged: (String) -> Unit,
    onTargetChanged: (String) -> Unit,
) {
    val sourceLanguages = remember(settings.speechEngine, settings.translationEnabled) {
        VoxlineLanguages.getFilteredLanguages(settings.speechEngine, settings.translationEnabled)
    }
    val targetLanguages = remember { VoxlineLanguages.targetLanguages() }

    Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
        Text(
            text = t("translation_desc"),
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        LanguageDropdown(
            label = t("source"),
            selectedTag = settings.sourceLanguageTag,
            languages = sourceLanguages,
            onSelected = onSourceChanged,
        )
        AnimatedVisibility(
            visible = settings.translationEnabled,
            enter = expressiveEnter(),
            exit = expressiveExit(),
        ) {
            LanguageDropdown(
                label = t("target"),
                selectedTag = settings.targetLanguageTag,
                languages = targetLanguages,
                onSelected = onTargetChanged,
            )
        }
        Surface(
            shape = MaterialTheme.shapes.medium,
            color = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.55f),
        ) {
            Text(
                text = if (settings.translationEnabled) {
                    t("translation_enabled_desc")
                } else {
                    t("translation_disabled_desc")
                },
                modifier = Modifier.padding(14.dp),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalMaterial3ExpressiveApi::class)
@Composable
private fun LanguageDropdown(
    label: String? = null,
    selectedTag: String,
    selectedLabel: String = VoxlineLanguages.labelFor(selectedTag),
    languages: List<VoxlineLanguage>,
    onSelected: (String) -> Unit,
) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = it },
        modifier = Modifier.fillMaxWidth(),
    ) {
        OutlinedTextField(
            value = selectedLabel,
            onValueChange = {},
            modifier = Modifier
                .fillMaxWidth()
                .menuAnchor(
                    type = ExposedDropdownMenuAnchorType.PrimaryNotEditable,
                    enabled = true,
                ),
            readOnly = true,
            singleLine = true,
            label = label?.let { dropdownLabel -> { Text(dropdownLabel) } },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            colors = ExposedDropdownMenuDefaults.outlinedTextFieldColors(),
        )
        ExposedDropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
            modifier = Modifier.heightIn(max = 320.dp),
        ) {
            languages.forEachIndexed { index, language ->
                val isSelected = language.tag == selectedTag
                DropdownMenuItem(
                    selected = isSelected,
                    onClick = {
                        expanded = false
                        onSelected(language.tag)
                    },
                    text = {
                        Text(
                            text = language.label,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                        )
                    },
                    shapes = MenuDefaults.itemShape(index = index, count = languages.size),
                    selectedLeadingIcon = if (isSelected) {
                        {
                            Icon(
                                painter = painterResource(R.drawable.sym_check_circle),
                                contentDescription = null,
                            )
                        }
                    } else {
                        null
                    },
                )
            }
        }
    }
}

@Composable
private fun UiLanguageSection(
    settings: VoxlineSettings,
    onLanguageSelected: (String) -> Unit,
) {
    val systemDefault = t("system_default")
    val languages = remember(systemDefault) {
        listOf(VoxlineLanguage(tag = "system", label = systemDefault)) + VoxlineLanguages.uiLanguages()
    }
    LanguageDropdown(
        selectedTag = settings.uiLanguageTag,
        selectedLabel = if (settings.uiLanguageTag == "system") {
            systemDefault
        } else {
            VoxlineLanguages.labelFor(settings.uiLanguageTag)
        },
        languages = languages,
        onSelected = onLanguageSelected,
    )
}

@Composable
private fun ThemeModeSection(
    settings: VoxlineSettings,
    onThemeModeSelected: (ThemeMode) -> Unit,
) {
    val themeModes = ThemeMode.entries.map { themeMode ->
        VoxlineLanguage(tag = themeMode.id, label = t(themeMode.labelKey))
    }
    LanguageDropdown(
        selectedTag = settings.themeMode.id,
        selectedLabel = t(settings.themeMode.labelKey),
        languages = themeModes,
        onSelected = { selectedId ->
            onThemeModeSelected(ThemeMode.fromId(selectedId))
        },
    )
}

private fun Context.hasPermission(permission: String): Boolean =
    checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED

private fun Context.overlaySettingsIntent(): Intent =
    Intent(
        Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
        "package:$packageName".toUri(),
    )

private fun Context.isVoxlineAccessibilityServiceEnabled(): Boolean {
    val captionService = ComponentName(this, VoxlineAccessibilityService::class.java)
    return getSystemService(AccessibilityManager::class.java)
        .getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_ALL_MASK)
        .any { service ->
            val serviceInfo = service.resolveInfo.serviceInfo
            ComponentName(serviceInfo.packageName, serviceInfo.name) == captionService
        }
}
