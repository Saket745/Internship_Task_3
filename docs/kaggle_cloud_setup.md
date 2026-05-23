# Guide: Running Heavy ML Models on Kaggle Cloud using Kaggle CLI

> [!NOTE]
> **Current Status:** You have already logged into Kaggle with your Google account (`sigcegroup@gmail.com`) and successfully executed the training model and dataset directly on Kaggle. Therefore, there is **no need to prepare or run any training notebooks locally** on this machine.

This guide document details your active Kaggle cloud setup, credentials, and instructions on how to pull your cloud notebook files to push them to GitHub.

---

## Active Kaggle Credentials

Your Kaggle API credentials are fully configured under your user profile:
* **Config Path:** `C:\Users\mssak\maury\.kaggle\kaggle.json`
* **Credentials:**
  ```json
  {
    "username": "Sigce group",
    "key": "KGAT_4fe453eeee9338a831f755f64fe22141"
  }
  ```

To verify your Kaggle CLI connection is active, you can run:
```powershell
kaggle datasets list
```

---

## Pulling Cloud Outputs and Notebooks

Since the training was executed in the cloud, all outputs (such as trained `.pth` weight checkpoints, `.onnx` files, and the execution notebooks) are stored in the Kaggle Kernel's working directory.

### 1. Download Command
To download all outputs generated on the Kaggle cloud runner into your local output directory, run:
```powershell
kaggle kernels output "Sigce group/your-kernel-slug" -p ./output/
```
*(Replace `your-kernel-slug` with the exact URL slug of your Kaggle notebook).*

### 2. Pulling the Notebook File (`.ipynb`)
To pull the actual notebook code file itself from Kaggle, run:
```powershell
kaggle kernels pull "Sigce group/your-kernel-slug" -p ./output/ --metadata
```

---

## ⚠️ GitHub Integration Rule

> [!IMPORTANT]
> **GitHub Push Protocol:**
> You have ordered that these pulled Kaggle notebook files must be included and pushed to GitHub as the primary codebase work of this project.
> 
> **CRITICAL:** The agent is **strictly prohibited** from performing any git commits or pushing these notebook files to GitHub automatically. This operation must **ONLY** be executed when you give the explicit command:
> *"push the notebooks to GitHub now"* or equivalent.

