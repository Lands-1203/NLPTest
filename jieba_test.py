import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 假设这是你的问题库
question_db = [
    ("我的瀚华工号是多少?", "你的瀚华工号是27897"),
    ("大x系统怎么给债务人设置债务减免?", "如果请按照一下步骤操作..."),
    ("大x系统怎么给债务人设置债务分期?", "如果请按照一下步骤操作..."),
    ("大x系统怎么登录?", "如果请按照一下步骤操作..."),
    ("瀚华系统怎么登录?", "如果请按照一下步骤操作..."),
    # 添加更多问题及答案
]

# 使用jieba进行分词


def chinese_tokenizer(text):
    return jieba.lcut(text)


# 初始化TF-IDF向量化器
vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer)

# 准备文本数据
texts = [q[0] for q in question_db]
# 训练TF-IDF模型
vectorizer.fit(texts)


def getAnswers(user_question):
    # 将用户问题转换为向量
    user_question_vec = vectorizer.transform([user_question])
    highest_similarity = 0
    best_match = None

    # 遍历问题库，找到最佳匹配
    for question, answer in question_db:
        question_vec = vectorizer.transform([question])
        similarity = cosine_similarity(user_question_vec, question_vec)[0][0]
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = (question, answer)

    print(f"用户问题: {user_question}")
    # 如果找到了相似度足够高的问题，返回答案
    if best_match and highest_similarity > 0.6:  # 假设阈值为0.6
        print(f"当前最匹配的问题: {best_match[0]}\n答案: {best_match[1]}")
    else:
        print("没有找到相关问题答案。")


getAnswers('登录瀚华系统')
getAnswers('我的工号是多少？')
getAnswers('登录大x系统')
getAnswers('怎么给债务人设置分期')
getAnswers('怎么给债务人设置减免')