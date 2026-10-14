"""Automated build script for MobileShield Android Companion App.

Uses local Android SDK build-tools (aapt2, d8, zipalign, apksigner) and JDK 21
to compile, dex, package, and sign the companion APK without requiring Gradle downloads.
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
COMPANION_DIR = PROJECT_ROOT / "android_companion"
BUILD_DIR = COMPANION_DIR / "build"
BIN_DIR = COMPANION_DIR / "bin"

# SDK Tools detection
LOCAL_APPDATA = os.environ.get("LOCALAPPDATA", r"C:\Users\Varun\AppData\Local")
SDK_DIR = Path(LOCAL_APPDATA) / "Android" / "Sdk"
BUILD_TOOLS_DIR = SDK_DIR / "build-tools" / "36.0.0"
PLATFORM_JAR = SDK_DIR / "platforms" / "android-37.0" / "android.jar"

AAPT2 = BUILD_TOOLS_DIR / "aapt2.exe"
ZIPALIGN = BUILD_TOOLS_DIR / "zipalign.exe"
D8_JAR = BUILD_TOOLS_DIR / "lib" / "d8.jar"
APKSIGNER_JAR = BUILD_TOOLS_DIR / "lib" / "apksigner.jar"

# JDK detection
JDK_BIN = Path(r"C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot\bin")
JAVA = JDK_BIN / "java.exe" if (JDK_BIN / "java.exe").exists() else Path("java")
JAVAC = JDK_BIN / "javac.exe" if (JDK_BIN / "javac.exe").exists() else Path("javac")
KEYTOOL = JDK_BIN / "keytool.exe" if (JDK_BIN / "keytool.exe").exists() else Path("keytool")


def run_cmd(cmd, desc):
    print(f"[*] {desc}...", flush=True)
    actual_cmd = [str(c) for c in cmd]
    env = os.environ.copy()
    if JDK_BIN.exists():
        env["JAVA_HOME"] = str(JDK_BIN.parent)
        env["PATH"] = f"{JDK_BIN};{env.get('PATH', '')}"
    res = subprocess.run(
        actual_cmd,
        cwd=str(COMPANION_DIR),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env
    )
    if res.returncode != 0:
        print(f"[!] Error in {desc}:", flush=True)
        print(res.stderr or res.stdout, flush=True)
        sys.exit(1)
    return res


def build():
    print("=" * 60)
    print("  BUILDING MOBILESHIELD ANDROID COMPANION APK")
    print("=" * 60)

    # Clean build dirs
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    gen_dir = BUILD_DIR / "gen"
    classes_dir = BUILD_DIR / "classes"
    dex_dir = BUILD_DIR / "dex"
    gen_dir.mkdir(exist_ok=True)
    classes_dir.mkdir(exist_ok=True)
    dex_dir.mkdir(exist_ok=True)

    # 1. Compile resources with aapt2
    compiled_res = BUILD_DIR / "compiled_res.zip"
    run_cmd([
        str(AAPT2), "compile", "--dir", "res", "-o", str(compiled_res)
    ], "Compiling XML resources")

    # 2. Link resources and generate R.java
    unsigned_apk = BUILD_DIR / "unsigned.apk"
    run_cmd([
        str(AAPT2), "link",
        "-I", str(PLATFORM_JAR),
        "--manifest", "AndroidManifest.xml",
        "-o", str(unsigned_apk),
        "--java", str(gen_dir),
        "--auto-add-overlay",
        str(compiled_res)
    ], "Linking APK package & generating R.java")

    # 3. Find and compile all Java sources
    java_files = list(COMPANION_DIR.glob("src/**/*.java")) + list(gen_dir.glob("**/*.java"))
    java_file_paths = [str(f) for f in java_files]

    run_cmd([
        str(JAVAC),
        "-cp", str(PLATFORM_JAR),
        "-d", str(classes_dir),
        "--release", "17"
    ] + java_file_paths, "Compiling Java sources with JDK 21")

    # 4. Dex compiled classes with d8
    class_files = [str(f) for f in classes_dir.glob("**/*.class")]
    run_cmd([
        str(JAVA), "-cp", str(D8_JAR),
        "com.android.tools.r8.D8",
        "--lib", str(PLATFORM_JAR),
        "--output", str(dex_dir)
    ] + class_files, "D8 dexing classes into classes.dex")

    # 5. Inject classes.dex into unsigned.apk
    dex_file = dex_dir / "classes.dex"
    print("[*] Adding classes.dex to APK archive...")
    with zipfile.ZipFile(unsigned_apk, "a") as zip_out:
        zip_out.write(dex_file, "classes.dex")

    # 6. Zipalign APK (4-byte alignment)
    aligned_apk = BUILD_DIR / "aligned.apk"
    run_cmd([
        str(ZIPALIGN), "-f", "-p", "4",
        str(unsigned_apk), str(aligned_apk)
    ], "Zipalign 4-byte optimization")

    # 7. Create debug keystore if not existing
    keystore = COMPANION_DIR / "debug.keystore"
    if not keystore.exists():
        run_cmd([
            str(KEYTOOL), "-genkeypair", "-v",
            "-keystore", str(keystore),
            "-storepass", "android",
            "-alias", "androiddebugkey",
            "-keypass", "android",
            "-keyalg", "RSA",
            "-keysize", "2048",
            "-validity", "10000",
            "-dname", "CN=MobileShield,O=AML,C=US"
        ], "Generating debug signing keystore")

    # 8. Sign APK with apksigner
    final_apk = BIN_DIR / "mobileshield_companion.apk"
    run_cmd([
        str(JAVA), "-jar", str(APKSIGNER_JAR), "sign",
        "--ks", str(keystore),
        "--ks-pass", "pass:android",
        "--out", str(final_apk),
        str(aligned_apk)
    ], "Signing APK with apksigner v2/v3")

    print("\n" + "=" * 60)
    print(f"[+] APK BUILD SUCCESS: {final_apk}")
    print(f"[*] Size: {final_apk.stat().st_size:,} bytes")
    print("=" * 60 + "\n")
    return str(final_apk)


if __name__ == "__main__":
    build()
