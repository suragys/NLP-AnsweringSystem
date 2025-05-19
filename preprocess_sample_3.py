import re
import skip_gram as sg
import numpy as np


class Preprocess(object):
    def __init__(self):
        self.inp = ""
        self.content = ""
        self.max_words_story = 0
        self.max_words_question = 0
        self.mat = []
        self.answer_words_lim = 16 - 1
        self.question_words_lim = 16 - 1
        self.story_words_lim = 220 - 1

    def read_input(self, trainfile, answerfile):
        f = open(trainfile, 'r')
        content = []
        for line in f:
            content.append(line.lower())

        f = open(answerfile, 'r')
        ans = ""
        for line in f:
            ans = ans + line

        self.content = content
        return content, ans


    def data_preprocess(self, content, ans):

        ip_for_SGM, data = self.generate_ip_for_SGM(content)

        story_ques_LSTM, answer_LSTM, story_LSTM, ques_LSTM, sq_lstm, a_lstm, s_lstm, q_lstm = self.extract_story_ques_for_LSTM(
            content)
        # labels is the label vector

        labels = self.create_labels_for_LSTM(ans)  


    def remove_all_punctuations(self, s):
        s = re.sub(r'[^\w\s]', ' ', s).lower()
        return s
        
    def isQ(self, line):
    	  if self.pattern_search('[1-4]:\ one:', line) == True or self.pattern_search('[1-4]:\ multiple:', line) == True:
    	      return True
    	  return False
    
    def isA(self, line):
        if line.startswith('a)') or line.startswith('b)') or line.startswith('c)') or line.startswith('d)'):
            return True
        return False
    
    def generate_ip_for_SGM(self, content):
        sgm_ip = ""
 

        for i in range(0, len(content)):
            line = content[i]
            line = line.strip()
            if line == '':
                continue
            s = " " + line

            if self.isA(line):
                temp = line.split(')')
                s = temp[1]
            if self.isQ(line):
                temp = line.split(':')
                s = temp[2]
            sgm_ip = sgm_ip + s
        sgm_ip = self.remove_all_punctuations(sgm_ip)
        data = sgm_ip.split()
        return sgm_ip, data

    def get_content(self, i):
        c = self.content[i].strip()
        return c

    def pattern_search(self, pattern, string):
        x = re.search(pattern, string)
        if x != None:
            return True
        else:
            return False

    def extract_story_ques_for_LSTM(self, content):
        answer_LSTM = []
        a_lstm = ""
        story_ques_LSTM = []
        sq_lstm = ""
        story_LSTM = []  
        s_lstm = ""  
        ques_LSTM = []  
        q_lstm = ""  
        cnt = 0

        for i in range(0, len(content)):
            line = self.get_content(i)

            if line.startswith('**'):
                # new section
                i += 1

                if self.get_content(i) == '':
                    cnt += 1
                    i += 1

                    story_words = []
                    tmps = ""
                    while self.isQ(self.get_content(i)) == False:                    
                        s = self.get_content(i)
                        s = self.remove_all_punctuations(s)
                        tmps += s
                        sq_lstm = sq_lstm + s
                        s_lstm += s

                        sw = s.split()
                        for w in sw:
                            story_words.append(w)
                        i += 1

                    story_ques_LSTM.append(tmps)
                    story_LSTM.append(tmps)
                    if len(story_words) > self.max_words_story:
                        self.max_words_story = len(story_words)
                        

                question_words = []
                if self.isQ(self.get_content(i)):
                
                    q_cnt = 0
                    while q_cnt < 4:
                        tmpq = self.get_content(i)
                        tq = tmpq.split(':')
                        q = tq[2]
                        story_ques_LSTM.append(self.remove_all_punctuations(q))
                        ques_LSTM.append(self.remove_all_punctuations(q))
                        question_words = q.split()

                        sq_lstm += q
                        q_lstm += q
                        i += 1
                        a_cnt = 0
                        while a_cnt < 4:
                            tmp = self.get_content(i)
                            t = tmp.split(')')
                            a = t[1]
                            answer_LSTM.append(self.remove_all_punctuations(a))
                            a_lstm += a
                            i += 1
                            a_cnt += 1
                        i += 1
                        q_cnt += 1

                if len(question_words) > self.max_words_question:
                    self.max_words_question = len(question_words)

        return story_ques_LSTM, answer_LSTM, story_LSTM, ques_LSTM, sq_lstm, a_lstm, s_lstm, q_lstm


    def create_labels_for_LSTM(self, ans):
        ans_vec = []
        for a in ans:
            for one_opt in a.split():
                if (one_opt == 'A'):
                    ans_vec.append([1.0, 0.0, 0.0, 0.0])
                    continue
                if (one_opt == 'B'):
                    ans_vec.append([0.0, 1.0, 0.0, 0.0])
                    continue
                if (one_opt == 'C'):
                    ans_vec.append([0.0, 0.0, 1.0, 0.0])
                    continue
                if (one_opt == 'D'):
                    ans_vec.append([0.0, 0.0, 0.0, 1.0])
        return ans_vec


    def get_emb_for(self, m, index, limit, sgm):
        start_symbol = np.ones(300)


        word_cnt = 0

        
        self.mat.append(start_symbol)

        for w in m[index]:

            try:
                index = sgm.dictionary[w]
                word_embedding = sgm.final_embeddings[index]
            except KeyError:
                word_embedding = np.zeros(300)
            self.mat.append(word_embedding)
            
            word_cnt += 1
            if word_cnt == limit:
            	break

        if word_cnt < limit:
            pad_num = limit - word_cnt
            zero = np.zeros(300)
            for i in range(0, pad_num):
                self.mat.append(zero)
                word_cnt += 1
        

    def get_ip_embedding_for_LSTM(self, ans, ques, story, a_limit, q_limit, s_limit, sgm):
        start_symbol = np.ones(300)
        s_index = 0
        q_index = 0
        a_index = 0
        while s_index < len(story):

            for q in range(0, 4):
                self.get_emb_for(story, s_index, s_limit, sgm)

                self.get_emb_for(ques, q_index, q_limit, sgm)
                for i in range(0, 4):
                    self.get_emb_for(ans, a_index, a_limit, sgm)

                    a_index += 1

                q_index += 1
            if len(self.mat) % 10 != 0:
        	      print("mistake here")                
            s_index += 1


    def get_ip_SQA_for_LSTM(self, sgm):
        stories = self.get_stories_new()
        
        questions = self.get_questions_new()
        
        answers = self.get_answers_new()
        self.get_ip_embedding_for_LSTM(answers, questions, stories, self.answer_words_lim, self.question_words_lim, self.story_words_lim, sgm)
        print( "last")
        print(len(self.mat))

        return self.mat

    def get_stories_new(self):
        story_ques_LSTM, answer_LSTM, story_LSTM, ques_LSTM, sq_lstm, a_lstm, s_lstm, q_lstm = self.extract_story_ques_for_LSTM(
            self.content)
        stories = []
        for s_arr in story_LSTM:
            s_words = s_arr.split()
            if len(s_words) > 0:
                stories.append(s_words)
        return stories

    def get_questions_new(self):
        story_ques_LSTM, answer_LSTM, story_LSTM, ques_LSTM, sq_lstm, a_lstm, s_lstm, q_lstm = self.extract_story_ques_for_LSTM(
            self.content)
        questions = []

        for q_arr in ques_LSTM:
            q_words = q_arr.split()
            if len(q_words) > 0:
                questions.append(q_words)
        return questions

    def get_answers_new(self):
        story_ques_LSTM, answer_LSTM, story_LSTM, ques_LSTM, sq_lstm, a_lstm, s_lstm, q_lstm = self.extract_story_ques_for_LSTM(
            self.content)
        answers = []
        for a_arr in answer_LSTM:
            a_words = a_arr.split()
            if len(a_words) > 0:
                answers.append(a_words)
        return answers
        
        


    def convert_to_indices(self, m1):
        indices_matrix = []
        for row_i in range(0, len(m1)):
            for word_i in row_i:
                index = sg.dictionary[m1[row_i][word_i]]
                indices_matrix[row_i][word_i] = index
        return indices_matrix


def main():
    obj = Preprocess()
    content, ans = obj.read_input("train_dev_data", "train_dev_ans")
    obj.data_preprocess(content, ans)


main()
