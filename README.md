# [AICUP 2024] Competition-2024-PyTorch-LLMRAG

💬 **Applications of RAG and LLM in Financial Q&A**  

## TEAM_6029: Kelvin, Jonathan, Edward, Tom   

---

- [**玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用**](https://tbrain.trendmicro.com.tw/Competitions/Details/37)  

<a href="https://tbrain.trendmicro.com.tw/Competitions/Details/37"><img src="https://i.imgur.com/54vUEr3.png" title="source: imgur.com" /></a>  
> 在大型語言模型加速催化各式技術的年代，語言模型的開發週期越來越短、效能越來越強。隨著大型語言模型的問世，金融業龐大且複雜的資料已經不再是語料檢索無法高度泛化的障礙，而是逐漸被解決的問題。
> 本屆挑戰賽聚焦在金融問答領域，提供豐富的資料庫供參賽者使用。參賽者需設計機制以提高檢索結果的準確性，包括從提供的語料中找出完整回答問題的正確資料等基本要求，以及應用大型語言模型的生成能力，產出正確且完整的回答。


<a href="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FFanChiMao%2FCompetition-2024-PyTorch-LLMRAG&label=visitors&countColor=%232ccce4&style=plastic" target="_blank">
  <img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FFanChiMao%2FCompetition-2024-PyTorch-LLMRAG&label=visitors&countColor=%232ccce4&style=plastic" alt="Visitors">
</a>

<a href="https://img.shields.io/github/downloads/FanChiMao/Competition-2024-PyTorch-LLMRAG/total" target="_blank">
  <img src="https://img.shields.io/github/downloads/FanChiMao/Competition-2024-PyTorch-LLMRAG/total" alt="Download">
</a>


## 📌 Quick Inference
### To reproduce our submit inference results, please following instructions.

<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 0: Environment Setting</b></span></summary>

  - **Download the Repo**
    ```commandline
    git clone https://github.com/FanChiMao/Competition-2024-PyTorch-LLMRAG.git
    cd Competition-2024-PyTorch-LLMRAG
    git submodule update --init
    ```
  
  - **Prepare the environment**  
    ❗ **Noted:** Please check your GPU and OS environment, and go to the [**PyTorch Website**](https://pytorch.org/get-started/previous-versions/) to install Pytorch first. 

    ```commandline
    conda create --name LLMRAG python=3.10  # to reproduce the results, you have to install python 3.10
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # take cuda 11.8 as example
    pip install -r requirements.txt
    ```
  
  <br>
  
</details>


<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 1: Preparing Datasets</b></span></summary>

  - Go to the [**official website**](https://tbrain.trendmicro.com.tw/Competitions/Details/37) to download the datasets. (due to the policy, we can't provide the dataset)

  - You can directly run the script
    ```commandline
    cd scripts
    1.download_preliminary_data.bat
    ```
    
    or run the snippet at [**./datasets/download_preliminary_datasets.py**](./datasets/download_preliminary_datasets.py)
    ```commandline
    cd datasets
    python ./download_preliminary_datasets.py
    ```
    
  - Place the dataset in [./datasets](datasets).  

  <br>
  
</details>


<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 2: Running Baseline</b></span></summary>

  - You can directly run the script to run the baseline code
    ```commandline
    cd scripts
    2.run_baseline_code.bat
    ```
    or run the snippet at [**./main_baseline.py**](main_baseline.py)
    
    ```commandline
    python ./main_baseline.py
    ```
    
  - After running the baseline code, it will generate the json result on [**./output/baseline.json**](outputs/baseline.json)

  <br>
  
</details>


<details>
  <summary><span style="font-size: 1.1em; vertical-align: middle;"><b>Step 3: Reproduce Results</b></span></summary>

  - To reproduce our submitted results, you can run 
    ```commandline
    cd scripts
    3.run_preliminary_results.bat
    ```
    or run the snippet at [**./main_preliminary.py**](./main_preliminary.py)
    
    ```commandline
    python ./preliminary_results.py
    ```

  - After running the baseline code, it will generate the json result on [**./output/preliminary_results.json**](outputs/preliminary_results.json)

  <br>
  
</details>


## 🕵️ Evaluation

### To evaluate the precision@1 for the output json, please following the command  

```commandline
python ./evaluation.py --gt [path of ground_truths_example.json] --rs [path of output json]
```

take baseline result for example:
```commandline
python ./evaluation.py --gt ./datasets/preliminary/ground_truths_example.json --rs ./outputs/baseline.json
```

<br>


## 📫 Contact Us
- **Kelvin**: [fxp61005@gmail.com]()  
- **Jonathan**: [qaz5517359@gmail.com]()  
- **Edward**: []()
- **Tom**: []()
