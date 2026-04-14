# О репозитории
Данный репозитория представляет собой решение домашнего задания курса NLP университета СПБГУ во втором семестре.

Расположение дз выглядит следующим образом:

```text
.
├── homework_1/                         # материалы первого домашнего задания
│   ├── results/
│   │   ├── tokenizer/                  # файлы токенизатора
│   │   ├── base_model_losses.png       # график loss при обучении базовой GPT
│   │   ├── gpt_pre-norm_state_dict.pt  # сохранённые параметры GPT
│   │   ├── gpt_pre-norm_test_losses.npy    # loss на test
│   │   └── gpt_pre-norm_train_losses.npy   # loss на train
│   ├── book.txt                        # текст, использованный для обучения модели
│   ├── gpt_dev.ipynb                   # ноутбук, предоставленный преподавателем
│   └── gpt_pre_norm.ipynb              # ноутбук с решением первого задания
│
└── homework_2/                         # материалы второго домашнего задания
    ├── results/
    │   ├── tokenizer/                                  # файлы токенизатора
    │   ├── GPTPostNorm_losses.png                      # график loss для GPT с post-norm
    │   ├── GPTPreNorm_losses.png                       # график loss для GPT с pre-norm
    │   ├── GPTPreNormWithRope_losses.png               # график loss для GPT с pre-norm и RoPE
    │   ├── GPTPreNormWithRopeWith_losses.png           # график loss для GPT с pre-norm, RoPE и MoE
    │   ├── gpt_post-norm_state_dict.pt                 # параметры GPT с post-norm
    │   ├── gpt_post-norm_test_losses.npy               # loss GPT с post-norm на test
    │   ├── gpt_post-norm_train_losses.npy              # loss GPT с post-norm на train
    │   ├── gpt_pre-norm_state_dict.pt                  # параметры GPT с pre-norm
    │   ├── gpt_pre-norm_test_losses.npy                # loss GPT с pre-norm на test
    │   ├── gpt_pre-norm_train_losses.npy               # loss GPT с pre-norm на train
    │   ├── gpt_pre-norm_with_RoPe_state_dict.pt        # параметры GPT с pre-norm и RoPE
    │   ├── gpt_pre-norm_with_RoPe_test_losses.npy      # loss GPT с pre-norm и RoPE на test
    │   ├── gpt_pre-norm_with_RoPe_train_losses.npy     # loss GPT с pre-norm и RoPE на train
    │   ├── gpt_pre-norm_with_RoPe_with_MoE_state_dict.pt      # параметры GPT с pre-norm, RoPE и MoE
    │   ├── gpt_pre-norm_with_RoPe_with_MoE_test_losses.npy    # loss GPT с pre-norm, RoPE и MoE на test
    │   ├── gpt_pre-norm_with_RoPe_with_MoE_train_losses.npy   # loss GPT с pre-norm, RoPE и MoE на train
    │   ├── models_test_losses.png                      # сравнение ошибок всех моделей на test
    │   └── models_train_losses.png                     # сравнение ошибок всех моделей на train
    ├── book.txt                                        # текст, использованный для обучения моделей
    ├── dataset_and_tokenizer.py                        # классы датасета и токенизатора
    ├── gpt_post_norm.py                               # реализация GPT с post-norm
    ├── gpt_pre_norm.py                                # реализация GPT с pre-norm
    ├── gpt_pre_norm_with_RoPe.py                      # реализация GPT с pre-norm и RoPE
    ├── gpt_pre_norm_with_RoPe_with_MoE.py             # реализация GPT с pre-norm, RoPE и MoE
    └── homework_2.ipynb                               # ноутбук с обучением и тестированием моделей
```
# Homework 1
В первом ДЗ я самостоятельно реализовал GPT с pre-norm (архитектура соответствует архитектуре, представленной в ноутбуке преподавателя).

Результаты обучения следующие:
![https://github.com/drema3-4/NLP-Homework-2-semester/blob/69b75069f44444255660e2c22ed66a74ef880206/homework%201/results/base_model_losses.png](https://github.com/drema3-4/NLP-Homework-2-semester/blob/69b75069f44444255660e2c22ed66a74ef880206/homework%201/results/base_model_losses.png)
