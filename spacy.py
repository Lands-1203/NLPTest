import spacy
from sklearn.metrics.pairwise import cosine_similarity

# 加载预训练的NLP模型
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("zh_core_web_sm")

# 假设这是你的问题库
question_db = [
    ("我的瀚华工号是多少?", "你的瀚华工号是27897"),
    ("大熵系统怎么给债务人设置债务减免?", "如果请按照一下步骤操作..."),
    ("大熵系统怎么给债务人设置债务分期?", "如果请按照一下步骤操作..."),
    ("大熵系统怎么登录?", "如果请按照一下步骤操作..."),
    ("瀚华系统怎么登录?", "如果请按照一下步骤操作..."),
    # 添加更多问题及答案
]
def getAnswers( user_question ):
    
    # 将用户问题转换为向量
    user_question_vec = nlp(user_question).vector.reshape(1, -1)
    # 初始化最佳匹配和最高相似度分数
    best_match = None
    highest_similarity = 0

    # 遍历问题库，找到最佳匹配
    for question, answer in question_db:
        question_vec = nlp(question).vector.reshape(1, -1)
        similarity = cosine_similarity(user_question_vec, question_vec)[0][0]
        print(similarity,question)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = (question, answer)


    print(f"当前最匹配的问题: {best_match[0]}")
    # 如果找到了相似度足够高的问题，返回答案
    if best_match and highest_similarity > 0.6:  # 假设阈值为0.5
        print(f"当前最匹配的问题: {best_match[0]}\n答案: {best_match[1]}")
    else:
        print("没有找到相关问题答案。")

getAnswers('登录瀚华系统')
getAnswers('我的工号是多少？')
getAnswers('登录大熵系统')
getAnswers('怎么给债务人设置分期')
getAnswers('怎么给债务人设置减免')