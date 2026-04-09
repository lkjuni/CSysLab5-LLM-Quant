# muduo

## 编译：  
```bash
cd src  
make clean  
make  
./muduo 模型路径 分词器路径 提示词
```

## 运行
1. Usage
```bash
muduo [model_path] [tokenizer_path] [prompt]  
```

2. 示例 
```bash
./muduo data/stories15M.bin data/tokenizer.bin "once upon a time,"
```