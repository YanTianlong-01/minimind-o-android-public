# MiniMind-O Android

MiniMind-O 的 Android 端实现：在手机上完全离线运行的多模态语音对话 App。音频输入 → 语音识别 → 大模型推理 → 语音合成，**全程在设备本地用 ONNX Runtime 完成，不依赖任何服务器**。

---

## 安全声明（为什么这个 App 是安全、无后门的）

这个 App 的全部源码都在本仓库公开，任何人可审计、可自行从源码构建。以下几点可验证 App 不含后门、不泄露隐私：

**1. 只申请一个权限。** 见 `app/src/main/AndroidManifest.xml`：

| 权限 | 用途 |
|---|---|
| `RECORD_AUDIO` | 录制麦克风音频，用于语音输入 |

仅此一项。**没有 `INTERNET`、没有 `ACCESS_NETWORK_STATE`、没有任何读写外部存储权限**（模型从 App 私有 assets 读取，不需要存储权限）。

**2. 不联网。** App 没有声明任何网络权限，代码里也没有任何 HTTP / socket / 上传逻辑。音频、文本、模型权重全部留在设备本地，**不可能把你的数据发到任何地方**。

**3. 不含后台组件。** `AndroidManifest.xml` 里只有一个 `MainActivity`（用户可见的界面），没有 `<service>`、`<receiver>`、`<provider>`。App 不做任何用户不可见的后台行为。

**4. 备份已关闭。** `android:allowBackup="false"`，App 数据不会被自动备份到 Google 服务器。

**5. 推理完全在本地。** 模型用 [ONNX Runtime](https://onnxruntime.ai/) 在设备端执行（见 `app/src/main/cpp/` 和 `app/src/main/java/com/minimind/phone/`），不调用任何云端 API。

如何自行验证：直接读 `app/src/main/AndroidManifest.xml`（权限清单）和全文搜索 `app/src/main/java/` 下所有源码——搜不到 `http`、`socket`、`upload`、`Internet` 等任何联网相关代码。

---

## App 界面

App 提供两个 tab：

- **User（默认）**：面向普通用户。Start/Stop Mic 置顶，删除了开发调试用的按钮，生成参数默认折叠，日志区文字长按可选中复制。
- **Developer**：完整开发者界面，含 prompt 文本运行、后端 smoke 测试、session 预热、parity benchmark、音频 golden 生成等调试入口。

两个 tab 共用同一个推理 runtime / 麦克风 / 音频播放器，日志同步。

---

## 从源码构建

### 环境要求

- Android Studio（任意较新版本，自带 JBR JDK 即可）
- Android SDK，`compileSdk 36` / `minSdk 26`
- NDK（用于编译 `app/src/main/cpp/` 下的 JNI 原生层）
- 构建机器架构不限：真机（arm64）和 x86_64 模拟器都支持（`abiFilters` 含 `arm64-v8a`、`x86_64`、`x86`）

### 步骤

1. 用 Android Studio `File` → `Open` 打开本仓库根目录（含 `settings.gradle` 的文件夹）。
2. 等 Gradle Sync 完成（首次会下载 Gradle、Android Gradle Plugin、ONNX Runtime 依赖）。
   - 国内若 `dl.google.com` TLS 握手失败，`settings.gradle` 里已配阿里云镜像。
   - 若提示 NDK 未安装，在 Android Studio 的 SDK Manager 里安装项目要求的 NDK 版本。
3. 选一个运行目标（真机或 x86_64 模拟器），点 Run。

构建产物：`app/build/outputs/apk/debug/app-debug.apk`。

> 关于模拟器架构：Windows 上的 Android 模拟器（QEMU2）只能运行和主机架构一致的镜像。Windows/Linux x86_64 主机请选 **x86_64** 系统镜像；`abiFilters` 已包含 x86_64 支持，无需改代码。

---

## 项目结构

```
app/
├── src/main/
│   ├── java/com/minimind/phone/      # Java 应用层：UI、推理 runtime、音频录制/播放、tokenizer
│   │   └── MainActivity.java         # 全部前端 UI（纯代码构建，非 XML）
│   ├── cpp/                          # JNI 原生层：minimind_runtime.cc 调用 ONNX Runtime
│   ├── assets/
│   │   ├── models/                   # 随 App 打包的小模型（VAD、tokenizer、audio projector）
│   │   └── demo/                     # 演示用的 golden 音频样本
│   └── AndroidManifest.xml           # 权限声明（仅 RECORD_AUDIO）
├── build.gradle                      # abiFilters、依赖、native build 配置
settings.gradle                       # Gradle 配置（含国内镜像）
gradle/ , gradlew, gradlew.bat        # Gradle wrapper（无需本机预装 Gradle）
```

> App 自带的 assets 模型（VAD、tokenizer、audio projector）只够跑 VAD + tokenizer + 音频投影的 smoke 流程。完整语音对话功能需要额外的 ONNX 大模型权重（GB 级），不随本仓库分发——需要自行从上游项目导出并 sideload 到设备。

---

## 技术栈

- **推理引擎**：ONNX Runtime (Android)，通过 JNI 调用
- **语音活动检测 (VAD)**：Silero VAD
- **语音识别 (ASR)**：SenseVoice encoder
- **多模态大模型**：MiniMind-O（Thinker decoder + Genner decoder）
- **神经编解码 / 语音合成**：Mimi
- **UI**：纯 Android View（Java，无 XML layout）

---

## 许可证

[MIT](LICENSE)。模型权重的许可以各自上游项目为准，不在本仓库 MIT 范围内。
