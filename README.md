

<div align="center">
  <img src="./assets/light.svg" alt="AgentCPM-Explore 标志" width="400em"></img>
</div>

<p align="center">
    【<a href="README_zh.md">中文</a> | English】
</p>



🚀 AgentCPM-Explore - open-source агент на 4B, который реально тащит GAIA и сложные реальные задачи  
✅ SOTA среди 4B агент-моделей  По агентным бенчмаркам модель:  
- обгоняет всех на своём масштабе  
- превосходит часть 8B моделей  
- и даже конкурирует с некоторыми 30B+ и closed-source LLM  
🧠 Deep Research как у “исследователя”  Модель умеет:  
- длинные цепочки рассуждений (long-horizon reasoning)  
- 100+ ходов автономного диалога  - проверять себя через несколько источников (cross-validation)  
- делать самокоррекцию как человек  
- динамически менять стратегию и использовать инструменты  То есть это уже не “чатбот”, а мини-исследователь, который реально может вести задачу до конца.  

🔓 Открыт не только модельный вес 
- открыт весь стек  И это самое жирное: OpenBMB выкладывают не “голую модель”, а весь pipeline агентности:  
- AgentRL - асинхронный RL-фреймворк для обучения агентов    
- AgentDock - безопасная песочница инструментов (tool sandbox)    
- AgentToLeaP - платформа оценки tool-learning (в один клик)    
- полный датапайплайн и воспроизводимые training workflows  

Это полноценная open-source платформа для создания агентов, где можно реально учиться, экспериментировать и собирать своих автономных “ресёрчеров”. 



# Latest News

* [2026-01-12] 🚀🚀🚀 We have open-sourced **AgentCPM-Explore**, the first open-source agent model with 4B parameters to appear on 8 widely used long-horizon agent benchmarks.

# Overview
AgentCPM is a series of open-source large language model agents jointly developed by the [THUNLP](https://nlp.csai.tsinghua.edu.cn), [Renmin University of China](http://ai.ruc.edu.cn/), [ModelBest](https://modelbest.cn/en), and [OpenBMB](https://www.openbmb.cn/en/home). 

# Model List

| Model            | Download Links                                                                                                                                | Resources | Technical Report | How to Use |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------|-----------|
| AgentCPM-Explore          | [🤗 Hugging Face](https://huggingface.co/openbmb/AgentCPM-Explore)<br> [🤖 ModelScope](https://modelscope.cn/models/OpenBMB/AgentCPM-Explore/)                  |  [AgentDock](./AgentCPM-Explore/AgentDock): An unified tool sandbox management and scheduling platform  <br> [AgentRL](./AgentCPM-Explore/AgentRL): An asynchronous agent reinforcement learning training framework  <br> [AgentToLeaP](./AgentCPM-Explore/AgentToLeaP): An one-click evaluation platform for agent tool-learning capabilities | Coming Soon | [README.md](./AgentCPM-Explore) |


## AgentCPM-Explore

The AgentCPM team has focused on systematically building agents’ deep research capabilities and released **AgentCPM-Explore**, a deep-search LLM agent. **AgentCPM-Explore** is the first open-source agent model with 4B parameters to appear on eight widely used long-horizon agent benchmarks such as GAIA, XBench, etc.

Key highlights:

- **SOTA at 4B Scale**: Best-in-class among same-size models, matches or surpasses 8B models, rivals some 30B+ and closed-source LLMs.

- **Deep Exploration**: 100+ turns of continuous interaction with multi-source cross-validation and dynamic strategy adjustment.

- **End-to-End Open Source**: Complete training and evaluation infrastructure for community development and custom extensions.


### Demo

Demo examples (speed up):

https://github.com/user-attachments/assets/f2b3bb20-ccd5-4b61-8022-9f6e90992baa


### QuickStart

- **Multi-model, multi-tool collaborative environment setup**: First, start the AgentDock tool sandbox platform to provide unified MCP (Model Context Protocol) tool services. When working with API-based models, configure the model’s `BASE_URL` and `API_KEY`. When working with locally hosted models, ensure the model service is accessible. Configure the required tool parameters in the `config.toml` file.

- **Launch the environment**: Out of the box, one-click startup. The AgentDock unified tool sandbox platform supports launching all services with a single `docker compose up -d` command, including the management dashboard, database, and tool nodes.

- **Run execution**: Quickly experience the core capabilities of the framework via the QuickStart script, allowing you to run a complete Agent task without complex configuration.

0. **Prepare Evaluation Environment (Recommended)**:  
   We provide a Docker image with all evaluation dependencies pre-installed. It is recommended to pull the image and run it directly:

   ```bash
   # 1. Enter the project folder
   cd AgentCPM-Explore
   
   # 2. Pull the image
   docker pull yuyangfu/agenttoleap-eval:v1.0
   
   # 3. Start the container (Adjust the -v path as needed)
   docker run -dit --name agenttoleap --gpus all --network host -v $(pwd):/workspace yuyangfu/agenttoleap-eval:v1.0
   
   # 4. Enter the container
   docker exec -it agenttoleap /bin/bash
   cd /workspace
   ```

1. **Configure and run**:  
  Open `quickstart.py` and make simple configurations in the `[USER CONFIGURATION]` section:

  - **Custom task**: Modify the `QUERY` variable to the instruction you want to test (e.g., “Check the results of last night’s UEFA Champions League matches”).
  - **Model information**: Provide your LLM `API_KEY`, `MODEL_NAME`, and `BASE_URL`.
  - **Tool service**: Set `MANAGER_URL` to the address of your MCP tool server (e.g., `http://localhost:8000`; make sure the service is already running).

  After configuration, run:

  ```bash
  python quickstart.py
  ```

  The script will automatically create a demo task (by default, querying today’s arXiv computer science papers), generate the execution workflow, and start the evaluation process.

2. **View Results**

  After execution completes, results will be saved under the `outputs/quickstart_results/` directory. You can inspect `dialog.json` to obtain the full interaction trace, including tool calls and reasoning chains.

  *Note: In QuickStart mode, automatic scoring is skipped by default and is intended only to demonstrate the Agent’s execution capabilities.*



# License

* The code in this repository is released under the [Apache-2.0](./LICENSE) license.

# Citation

If **AgentCPM-Explore** is useful for your research, please cite the codebase:

```bibtex
@software{AgentCPMExplore2026,
  title  = {AgentCPM-Explore: An End-to-End Infrastructure for Training and Evaluating LLM Agents},
  author = {Haotian Chen and Xin Cong and Shengda Fan and Yuyang Fu and Ziqin Gong and Yaxi Lu and Yishan Li and Boye Niu and Chengjun Pan and Zijun Song and Huadong Wang and Yesai Wu and Yueying Wu and Zihao Xie and Yukun Yan and Zhong Zhang and Yankai Lin and Zhiyuan Liu and Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM}
}
```

# Explore More

- [AgentCPM-GUI](https://github.com/OpenBMB/AgentCPM-GUI)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)
- [UltraRAG](https://github.com/OpenBMB/UltraRAG)
