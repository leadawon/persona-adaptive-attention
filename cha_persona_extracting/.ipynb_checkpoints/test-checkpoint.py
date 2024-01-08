# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional
from ast import literal_eval
import fire
import sys
import json
import jsonlines
from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    final=[]
    count=-1
    cur_dial=[]
    session1=[]
    session2=[]
    session3=[]
    current=[]
    print("start------")
    with jsonlines.open("./msc/test2.jsonl",'r') as f:
        for line in f:
            #session1.append(line['session1'])
            #session2.append(line['session2'])
            sen=""
            sen2=""
            sen3=""
            sen4=""
            for sess1 in line['session1']:
                sen+=sess1+"\n"
            for sess2 in line['session2']:
                sen2+=sess2+"\n"
            for sess3 in line['session3']:
                sen3+=sess3+"\n"
            current.append(line['session4'])
                
            user1={
                "role": "user", 
                "content": "Extract each speaker's personas based on the dialogue.\nThe extracted personas should be the sentence like subject is related with the object.\nDialogue:\nSpeaker 1: Good morning, how are you feeling today.\nSpeaker 2: Good morning, I am fine, how are you?\nSpeaker 1: Great, I am listening to different types of before I go for a bike ride.\nSpeaker 2: Sounds great! I am off to college in a bit. Full schedule and then summer classes too.\nSpeaker 1: Awesome I am a computer genius and love reading.\nSpeaker 2: Wow. Impressive. I am into martial arts. Jiu-jitsu. I instruct that.\nSpeaker 1: I eat lots of meats plants to healthy.\nSpeaker 2: That is healthy. I stay away from dairy. I am allergic.\nSpeaker 1: Do you like to eat coconut based products.\nSpeaker 2: I like coconut...yes! Where is your location?\nSpeaker 1: I live in new york.\nSpeaker 2: Same here!!! Pretty dreary again today.\n"
            }
            assist={
                "role": "assistant",
                "content":"\nSpeaker 1's persona:\n\n* is listening to different types.\n* is computer genius.\n* loves reading.\n* eat lots of meats plants.\n* lives in new york.\nSpeaker 2's persona:\n\n* are off to college in a bit.\n* are into martial arts called Jiu-jitsu.\n* has allergic of dairy\n* likes cocunut\n* lives new york"
            }
            user2={"role": "user", "content": "Extract each speaker's personas based on the dialogue.\nThe extracted personas should be the sentence like subject is related with the object.\nDialogue:\nSpeaker 1: I need some advice on where to go on vacation, have you been anywhere lately?\nSpeaker 2: I have been all over the world. I'm military.\nSpeaker 1: That is good you have alot of travel experience.\nSpeaker 2: Sure do. And a lot of experience blowing things up! Haha. Bora bora is nice.\nSpeaker 1: I've been working non stop crazy hours and need a break.\nSpeaker 2: The best breaks are spent with cute cuddly kittens.\nSpeaker 1: Bora bora sounds nice, you have been there before?\nSpeaker 2: Nope... Just sounds nice, and repetitive. Bora... Bora. Ha!\nSpeaker 1: Kittens really? I rather be at the beach.\nSpeaker 2: Only if the beach was covered in kittens!\nSpeaker 1: That would be a sight to see.\nSpeaker 2: Or maybe brownies... I love chocolate.\nSpeaker 1: I love brownies too but I haven't quite perfected mine yet.\nSpeaker 2: Well I'm available to taste test!\n"}
            assist2={
                "role": "assistant",
                "content":"\nSpeaker 1's persona:\n\n* needs advice on vacation.\n* has been working non-stop with crazy hours.\n* prefers the beach.\n* loves brownies.\nSpeaker 2's persona:\n\n* has been all over the world. (military background)\n* has experience blowing things up.\n* likes cute and cuddly kittens\n* prefers repetitive and exotic destinations. (Bora Bora)\n* loves chocolate (brownies)\n\n"
            }
            user3={}
            user4={}
            user5={}
            user3["role"]="user"
            user3["content"]="Extract each speaker's personas based on the dialogue.\nThe extracted personas should be the sentence like subject is related with the object.\nAnswer in the same form as in the previous example.\nDialogue:\n"+sen
            user4["role"]="user"
            user4["content"]="Extract each speaker's personas based on the dialogue.\nThe extracted personas should be the sentence like subject is related with the object.\nAnswer in the same form as in the previous example.\nDialogue:\n"+sen2
            user5["role"]="user"
            user5["content"]="Extract each speaker's personas based on the dialogue.\nThe extracted personas should be the sentence like subject is related with the object.\nAnswer in the same form as in the previous example.\nDialogue:\n"+sen3
            
            sample1=[user1,assist,user2,assist2,user3]
            sample2=[user1,assist,user2,assist2,user4]
            sample3=[user1,assist,user2,assist2,user5]
            final.append([sample1,sample2,sample3])
    print(len(final))
 
    persona_list=[]
    for i in range(len(final)):
        results = generator.chat_completion(
            final[i],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        for dialog, result in zip(final[i], results):
            #for msg in dialog:
            #    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            #print(
            #    f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            #)
            persona_list.append(result['generation']['content'])
    #print(len(persona_list))
            #print("\n==================================\n")
    with open("session4_persona_list.txt",'w') as f:
        #index=0
        for t in range(0,len(persona_list),3):
            f.writelines("--session 1 persona list--: \n")
            f.writelines(persona_list[t]+'\n')
            f.writelines("--session 2 persona list--: \n")
            f.writelines(persona_list[t+1]+'\n')
            f.writelines("--session 3 persona list--: \n")
            f.writelines(persona_list[t+2]+'\n')
            f.writelines("--current dialogue--: \n")
            f.writelines(str(current[t//2])+"\n")
            #index+=1


if __name__ == "__main__":
    fire.Fire(main)