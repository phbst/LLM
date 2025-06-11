import sentencepiece as spm
import os

def preprocess_file(input_file, output_file):
    story_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in:
        content = f_in.read()
        story_count = content.count('<|endoftext|>')
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in content.split('\n'):
                line = line.replace("<|endoftext|>", "\n")
                f_out.write(line + '\n')

    print(f"统计结果:")
    print(f"- 总共包含 {story_count} 个 story 段落")
    print(f"- 已将处理后的文本保存到 {output_file}")

def train_vocab(tiny_file, prefix, vocab_size):
    spm.SentencePieceTrainer.train(
            input=tiny_file,        
            model_prefix=prefix,     
            model_type="bpe",        
            vocab_size=vocab_size,   
            self_test_sample_size=0,
            input_format="text",     
            character_coverage=1.0,  
            num_threads=os.cpu_count(),  
            split_digits=True,      
            allow_whitespace_only_pieces=True,  
            byte_fallback=True,     
            unk_surface=r" \342\201\207 ",  # UNK token 表示未知字符的方式
            normalization_rule_name="identity"
    )

if __name__=='__main__':
    original_file = '/workspace/projects/Bob_llama/data/TinyStories_small.txt'
    processed_file = './data/TinyStories_processed.txt'
    prefix = 'tokens_4096'
    vocab_size = 4096
    
    # 预处理文件
    preprocess_file(original_file, processed_file)
    
    # 添加用户确认步骤
    user_input = input("是否要进行分词器训练? (Y/N): ").strip().upper()
    
    if user_input == 'Y':
        print("开始训练分词器...")
        # 使用预处理后的文件训练分词器
        train_vocab(processed_file, prefix, vocab_size)
        print("分词器训练完成!")
    else:
        print("已取消分词器训练")