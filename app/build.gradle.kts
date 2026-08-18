plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.jeremysu0818.voxline"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.jeremysu0818.voxline"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-O3")
                arguments += listOf("-DANDROID_STL=c++_shared")
                targets += "voxline_nemotron"
            }
        }

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
    packaging {
        jniLibs {
            // ggml discovers backend modules from applicationInfo.nativeLibraryDir at runtime,
            // so keep JNI libraries as extracted files instead of APK-zip-backed mappings.
            useLegacyPackaging = true
            // Whisper and the bundled Nemotron runtime both use the same Android C++ runtime.
            pickFirsts += "lib/**/libc++_shared.so"
            // Build-time ABI stub only. At runtime libggml-opencl resolves the device's
            // public libOpenCL.so through the Android linker namespace.
            excludes += "lib/**/libOpenCL.so"
        }
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.28.3"
        }
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.mlkit.genai.speech)
    implementation(libs.mlkit.translate)
    implementation(libs.opencc.java)
    implementation(libs.whisper.android)
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}
