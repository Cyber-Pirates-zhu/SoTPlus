from typing import List, Dict
from SoT_plus.prompter import Prompter


class s_Prompter(Prompter):

    def Initialize_CoT_prompt(self, question: str) -> str:
        initialize_cot_prompt = """
                <Instruction>
                    Split a math problem into two parts, one is the known information of the problem, and the other is the problem required to be solved.
                    
                    Finally, a dictionary needs to be returned in the format of: 
                    {"known information": A list of strings, each string represents a known information, "target": Problems to be solved in the question}
                    Note that each known information should contain only one known quantity or one known piece of information in the question. For some sentences with multiple known information, the known information should be separated into separate sentences. 
                    Each known information should fully represent the information of the known quantity, such as time, place, the item referred to by the quantifier, etc.
                    For example, "Yesterday morning, she got $5 and in the evening, she got $8." should be separated into two sentences: "Tom earned $5 in yesterday morning." and "Tom earned $8 in yesterday morning."
                </Instruction>
                
                <Examples>
                    Input: Tina buys 3 12-packs of soda for a party. Including Tina, 6 people are at the party. Half of the people at the party have 3 sodas each each, 2 of the people have 4, and 1 have has 5. How many sodas are left over when the party is over?
                    Output: {"known information": ["Tina buys 3 12-packs of soda for a party.", "6 people are at the party.", "Half of the people at the party have 3 soda.", "2 of the people at the party have 4 soda.", "1 people at the party have has 5 soda."], "target": "How many sodas are left over when the party is over?"}
                    
                    Input: Today, Tom earned $10, $21, and $6 in the morning, afternoon, and evening respectively. How much money did Tom earn in total today?
                    Output: {"known information": ["Today, Tom earned $10 in the morning.", "Today, Tom earned $21 in the afternoon.", "Today, Tom earned $6 in the evening."], "target": "How much money did Tom earn in total today?"}
                </Examples>
                
                Split the following problem into known information and question:
                <Input>{input}</Input>
                Please return a dictionary directly as in the examples, without any extra text.
                """
        return initialize_cot_prompt.replace('{input}', f"{question}")

    def generate_key_goal_prompt(self, known_information: list[str], additional_known_information: List[str],
                                 goal: str) -> str:
        Generate_key_goal_prompt = """
                <Instruction>
                    The input is a math problem. Note that all quantities expressed with symbols in the description of mathematical problems, such as x1, are known quantities and do not need to be solved. 
                    If you are a math teacher, please refer to the pattern in the example to generate an intermediate sub-problem for this problem to guide students to solve the final answer.
                </Instruction>

                <Examples>
                    Input: Peter bought 3 burgers. John bought 6 burgers. Lisa bought twice the total number of burgers that Peter and John bought combined. How many burgers did Lisa buy?
                    Output: How many burgers did Peter and John buy in total?

                    Input: There are 5 candies in box A. There are 6 candies in box B. The number of candies in box C is twice as many as the number of candies in box A and box B combined. 
                           The number of candies in box D is 2 less than that in box C. There are x1 candies in box A and box B in total. How many candies are in box D?
                    Output: How many candies are in box C?

                    Input: There are 3 fish in the iron bucket. There are 5 fish in the wooden bucket. The number of fish in the plastic bucket is 3 times the number of fish in the wooden bucket plus the iron bucket.
                           The number of fish in the net is 5 times the number of fish in the iron bucket plus the wooden bucket. There are x1 fish in the iron bucket and the wooden bucket.
                           There are x2 fish in the plastic bucket. How many fish are there in the net and the plastic bucket?
                    Output: How many fish are in the net?
                </Examples>

                Please generate a sub-problem for the following math problem to guide students to find the final answer.
                <Input>{input}</Input>
                Please return a question directly as in the examples, without any extra text.
                """
        Known_information = known_information + additional_known_information
        Input = " ".join(Known_information) + " " + goal
        return Generate_key_goal_prompt.replace('{input}', f"{Input}")

    def Create_known_information(self, sub_questions: list[str]) -> str:
        create_known_conditions = """
                <Instruction>
                    The input is an array of strings, each string in the array represents a question. 
                    Please refer to the method in the example to convert each question into a known information and use a variable, such as x1, 
                    to represent the quantity in the questions.
                    
                    You should output an list of dictionaries, where each dictionary has two keys. 
                    One key is called "question", which stores the problem before it is transformed. 
                    The other key is called "known information", which stores the known information after the problem is transformed.
                </Instruction>

                <Examples>
                    Input:  [
                                "Calculate how many sodas Tina bought?",
                                "Calculate how many sodas are occupied?",
                                "How many people have 3 sodas?"
                            ]
                    Output: [
                                {"question": "Calculate how many sodas Tina bought?", "known information": "Tina bought x1 sodas in total"},
                                {"question": "Calculate how many sodas are occupied?", "known information":"A total of x1 sodas were occupied"},
                                {"question": "How many people have 3 sodas?", "known information":"There are x1 people have 3 sodas"}
                            ]
                </Examples>

                Generate known information based on the following input questions
                <Input>{input}</Input>
                Please return a list of dictionary directly as in the examples, without any extra text.
        """
        return create_known_conditions.replace('{input}', f"{sub_questions}")

    def Logic_dependency_detection_prompt(self, known_information: list[str], additional_known_information: list[str],
                                          question: str) -> str:
        logic_dependency_detection_prompt = """
                <Instruction>
                    The input is a dictionary type data. The "known information" in the dictionary store some known information, and the "question" stores a question.
                    Note that all quantities expressed with symbols in the known information, such as x1, are known quantities and do not need to be solved. 
                    
                    Please determine whether you can answer the question in "question" based on the information in "known information".

                    Finally, you need to output a dictionary. There are two keys in the dictionary. The "result" key stores a Boolean value. 
                    True means that the answer to the question can be solve based on the known information, and False means that the question cannot be solve based on the known information.
                    The "explain" key stores the basis for the judgment.
                </Instruction>

                <Examples>
                    Input:{"known information": "The Chinese football team has accumulated 6 points. The South Korean football team has accumulated 5 points.", "question": "How many points did the Japanese team accumulate?"}
                    Output:{"explain": "The question does not mention any information about the Japanese team. So it is impossible to answer the corresponding question based on the known information.", "result": False}
                    
                    Input:{"known information": "Mary has 5 skirts. Li Lei has 3 skirts. The number of skirts that Lisa has is the sum of Mary and Li Lei.", "question": "How many skirts does Lisa have?"}
                    Output:{"explain": "The known information states that the number of shirts that Lisa has is the sum of Mary and Li Lei. So according to the known information, we can solve the corresponding problem", "result": True}
                    
                    Input:{"known information": "At a party, 6 people have 5 Easter eggs, and 4 people have 7 Easter eggs.", "question": "How many people have 2 Easter eggs?"}
                    Output:{"explain": "There is no known information that mentions anyone owning two Easter eggs. So it is impossible to answer the corresponding question based on the known information.", "result": False}
                </Examples>

                Determine whether the known information in "known information" can solve the question in "question".
                <Input>{input}</Input>
                Please return a dictionary directly as the examples, without any extra text.
                """
        Known_information = known_information + additional_known_information
        Known_information = " ".join(Known_information)
        Input = {"know information": Known_information, "question": question}
        return logic_dependency_detection_prompt.replace('{input}', f"{Input}")

    def Direct_logical_duplicate_detection_prompt(self, known_information: list[str], additional_known_information: List[str],
                                                  question: str) -> str:
        direct_logical_duplicate_detection_prompt = """
                <Instruction>
                    The input is a dictionary type data. The "known information" in the dictionary store some known information, and the "question" stores a question.
                    Note that all quantities expressed with symbols in the known information, such as x1, are known quantities and do not need to be solved. 
                    
                    Determine whether calculations are required to get the answer to the question based on the known information.

                    Finally, you need to output a dictionary. There are two keys in the dictionary. The "result" key stores a Boolean value. 
                    True means that calculation is required to get the answer to the question based on the known information. False means that it is not required.
                    The "explain" key stores the basis for the judgment.
                </Instruction>

                <Examples>
                    Input:{"known information": "There are 5 eggs in basket A. There are 10 eggs in basket B. The number of eggs in basket C is 2 more than the sum of the eggs in baskets A and B. There are x1 eggs in baskets A and B.", "question": "How many eggs are there in baskets A and B?"}
                    Output:{"explain": "The known information mentions 'There are x1 eggs in baskets A and B.'. So without any calculations we can get the final answer. ", "result": False}
                    
                    Input:{"known information": "There are 5 eggs in basket A. There are 10 eggs in basket B. The number of eggs in basket C is 2 more than the sum of the eggs in baskets A and B. There are x1 eggs in baskets A and B.", "question": "How many eggs are there in baskets C?"}
                    Output:{"explain": "The known information mention 'The number of eggs in basket C is 2 more than the sum of the eggs in baskets A and B.'. So we need to calculate (5+10+2) to get the answer to the question. ", "result": True}
                    
                    Input:{"known information": "There are 10 mails in the mailbox. Tina picked out 5 mails in the morning. Tina picked out 3 mails in the afternoon. Tina took x1 mails in total", "question": "How many emails have been taken from the mailbox?"}
                    Output:{"explain": "The known information says 'Tina took x1 mails in total.', this is equivalent to directly stating how many letters have been taken from the mailbox. without any calculations we can get the final answer. ", "result": False}
                </Examples>

                Determine whether calculations are required to get the answer to the question based on the known information.
                <Input>{input}</Input>
                Please return a dictionary directly as the examples, without any extra text.
                """
        Known_information = known_information + additional_known_information
        Known_information = " ".join(Known_information)
        Input = {"know information": Known_information, "question": question}
        return direct_logical_duplicate_detection_prompt.replace('{input}', f"{Input}")

    def Indirect_logical_duplication_detection_prompt(self, known_information: List[str],
                                                      additional_known_information: List[str],
                                                      information: str,
                                                      goal: str) -> str:
        indirect_logical_duplication_detection = """
                <Instruction>
                    The input data is a dictionary. The "question" key stores a math problem. The "information" key stores a piece of information related to the math problem.
                    Note that all quantities expressed with symbols in the known information, such as x1, are known quantities and do not need to be solved. 
                    
                    Please determine whether the information provided in "information" can simplify the process of solving the math problem in "question".

                    Finally, you need to output a dictionary. There are two keys in the dictionary. The "result" key stores a Boolean value. 
                    True means that the information can simplify the process of solving the math problem in "question", False means it cannot.
                    The "explain" key stores the basis for the judgment.
                </Instruction>

                <Examples>
                    Input:{"question": "Peter has 5 dollars. Rock has 6 dollars. Lisa has 3 times as much money as Peter and Rock combined. Max has 10 dollars more than Lisa. Lisa has x1 dollars. How much money does Max have?", "information": "Peter and Rock have x2 dollars in total."}    
                    Output:{"explain": "In the question, we know that Lisa has x1 dollars in total, and Max has 10 dollars more than Lisa. We only need to add x1 + 10 to get the answer. We don't need to know how much money Peter and Rock have in total. So this information cannot simplify the solution of the problem.", "result": False}
                    
                    Input:{"question": "Tina earned $100 yesterday morning and $120 in the afternoon. Tina earned $10 more today than yesterday. How much did Tina earn today?", "information": "Tina earned a total of $x1 yesterday."}    
                    Output:{"explain": "The question mentioned that Tina earned $10 more today than yesterday. Based on the information mentioned in 'information', we only need to calculate the result of x1 + 10 to get the final answer, which simplifies the calculation of how much money Tina earned yesterday.", "result": True}
                </Examples>

                Please determine whether the information provided in "information" can simplify the process of solving the math problem in "question".
                <Input>{input}</Input>
                Please return a dictionary directly as the examples, without any extra text.
                """
        Known_information = known_information + additional_known_information
        Known_information = " ".join(Known_information) + goal
        Input = {"question": Known_information, "information": information}
        return indirect_logical_duplication_detection.replace('{input}', f"{Input}")

    def score_prompt(self, known_information: list[str], additional_known_information: List[str], information: List[str],
                     goal: str) -> str:
        Score_prompt = """
                <Instruction>
                    The input data is a dictionary. The "question" key stores a math problem. The key "information" is an array of strings, each string is a piece of supplementary information to solve the math problem.
                    Note that all quantities expressed with symbols in the known information, such as x1, are known quantities and do not need to be solved. 
                    
                    Please rate each piece of supplementary information in "information". The lowest score is 0 and the highest score is 10. The more helpful the information is in solving the problem, the higher the score will be, and vice versa.
                    
                    Finally, you need to generate a list of dictionaries. Each dictionary in the list has three keys. 
                    The key "information" stores the information corresponding to the key "information" in the input dictionary. 
                    The key "result" stores the scoring results. The key "explain" stores the explanation of the scoring results.
                </Instruction>

                <Examples>
                    Input:{"question": "Peter has 5 dollars. Rock has 6 dollars. Lisa has 3 times as much money as Peter and Rock combined. Max has 10 dollars more than Lisa. How much money does Max have?", "information": ["Peter and Rock have x2 dollars in total.", "Lisa have x3 dollars in total."]}    
                    Output:[
                                {"information": "Peter and Rock have x2 dollars in total.", "explain": "This information can simplify the calculation of the amount of money Lisa has.", "result": 8}
                                {"information": "Lisa have x3 dollars in total.", "explain": "In the question, Max has 10 dollars more than Lisa. We only need to add x1 + 10 to get the answer. This information greatly simplifies the computational process of solving the problem", "result": 10}
                           ]                
                </Examples>

                Please rate each piece of information in "information" according to how helpful it is in solving the problem in "question".
                <Input>{input}</Input>
                Please return a list of dictionaries directly as the examples, without any extra text.
                """
        Known_information = known_information + additional_known_information
        Known_information = " ".join(Known_information) + goal
        Input = {"question": Known_information, "information": information}
        return Score_prompt.replace('{input}', f"{Input}")

    def decomposition_necessity_prompt(self, known_information: list[str], additional_known_information: list[str],
                                       goal: str) -> str:
        Decomposition_necessity_prompt = """
                <Instruction>
                    The input is a math problem. Note that all quantities expressed with symbols in the known information, such as x1, are known quantities and do not need to be solved. 
                    
                    Your task is to determine whether you can use only one basic operation such as addition, subtraction, multiplication and division to get the answer to the problem based on the known information of the math problem.
                    
                    Finally, you need to output a dictionary. There are two keys in the dictionary. The "result" key stores a Boolean value. 
                    True means that based on the known information of the math problem, the problem can be solved within one basic operation, and False means that the problem cannot be solved within one step.
                    The "explain" key stores the basis for the judgment. Note, the explanation needs to list the entire formula to solve the problem and count how many operators there are. Please refer to the answer in the example to generate the explanation.
                </Instruction>

                <Examples>
                    Input: Peter has 5 dollars. Rock has 6 dollars. Lisa has 3 times as much money as Peter and Rock combined. Max has 10 dollars more than Lisa. Lisa have x3 dollars in total. How much money does Max have?
                    Output: {"explain": "We only need to calculate x3 + 10 to get the final answer. This formula (x3 + 10) has only one mathematical symbol '+'. So the problem can be solved in one basic calculation.", "result": True}
                    
                    Input: Peter has 5 dollars. Rock has 6 dollars. Lisa has 3 times as much money as Peter and Rock combined. Max has 10 dollars more than Lisa. Lisa have x3 dollars in total. Peter and Rock have x2 dollars in total. How much money does Max have?
                    Output: {"explain": "We need to calculate x2 * 3 + 10 to get the final answer. This formula (x2 * 3 + 10) has two mathematical symbol, that is '*' and '+'. So the problem cannot be solved in one basic calculation, it need two basic calculation.", "result": False}              
                </Examples>

                Based on the known information of the following math problem, please determine whether the math problem can be solved in one operation.
                <Input>{input}</Input>
                Please return a dictionary directly as the examples, without any extra text.
                """
        Known_information = known_information + additional_known_information
        Input = " ".join(Known_information) + goal
        return Decomposition_necessity_prompt.replace('{input}', f"{Input}")

    def subtask_classification_prompt(self, known_conditions: str, additional_known_conditions: List[str],
                                      goals: str) -> str:
        Subtask_classification_prompt = """
                <Instruction>
                    The input is a dictionary type data, representing a problem. The "know conditions" key stores the known conditions of the problem, 
                    the "target" key stores the target to be solved and the key "additional known conditions" stores the intermediate quantities or 
                    additional known conditions that have been solved.
                    According to the known conditions corresponding to the "know conditions" key and the "addition known conditions" key 
                    in the dictionary and the final goal to be completed in the key "target", the task is judged to belong to one of the 
                    following three categories. Category 1: "math", which represents mathematical tasks. Category 
                    2: "text", which represents text generation tasks. Category 
                    3: "algorithm", which represents tasks suitable for completion using code.
                    Please return a string directly, such as "algorithm", without any extra text.
                </Instruction>

                <Examples>
                    Input:{"known conditions": "Tina buys 3 12-packs of soda for a party. Including Tina, 6 people are at the party.
                    Half of the people at the party have 3 sodas each each, 2 of the people have 4, and 1
                    have has 5.", "additional known conditions": ["Tina bought x1 packs of soda in total", "There are x2 people have 3 packs of soda"], 
                    "target": "How many sodas does the person who owns 3 sodas have in total?"}
                    Output: "math"
                    
                    Input:{"know conditions": "Tina buys 3 12-packs of soda for a party. Including Tina, 6 people are at the party.
                    Half of the people at the party have 3 sodas each each, 2 of the people have 4, and 1
                    have has 5. How many sodas are left over when the party is over?", "target":"Extract known conditions and issues for this problem."}
                    Output: "text"
                    
                    Input:{"know conditions": "Given an array [4, 3, 7, 8, 10, 3, 1].", "target": "Sort the elements in an array in ascending order."}
                    Output: "algorithm"
                </Examples>

                Determine the type of the problem based on the following known conditions:
                <Input>{input}</Input>
                """
        Input = {"know conditions": known_conditions, "additional known conditions": additional_known_conditions,
                 "target": goals}
        print("subtask_classification_prompt: ", Input)
        return Subtask_classification_prompt.replace('{input}', f"{Input}")

    def dependency_detection_prompt(self, known_information: list[str], sub_target: str, goal: str) -> str:
        Dependency_detection_prompt = """
                <Instruction>
                    The input is a dictionary. The dictionary key "question" stores a math problem. 
                    The dictionary key "Process quantity" stores a known information. 
                    Note that the symbolic quantity in the known information, such as x1, should be treated as a known quantity.
                    
                    Please determine whether it is necessary to solve the process quantity in "Process quantity" in the process of solving the mathematical problem in "question".
                    
                    Finally, you need to return a dictionary. The dictionary should have two keys, "explain" and "result".
                    The "explain" key stores a string that explains the result of the judgment.
                    True means that the process quantity in "Process quantity" needs to be solved in the process of solving the math problem in "question", False means it is not necessary.
                </Instruction>

                <Examples>
                    Input:{"question": "John has 5 cards. Lisa has 6 cards. Tina has 3 more cards than John and Lisa combined. Li Lei has 7 more cards than John. How many cards does Li Lei have in total?", 
                    "Process quantity": "Tina has a total of x1 cards."}
                    Output: {"explain": "Based on the known information, to know the number of cards Li Lei has, it is not necessary to first solve the number of cards Tina has. So the information in 'Process quantity' is not necessary", "result": False}
                    
                    Input:{"question": "John has 5 cards. Lisa has 6 cards. Tina has 3 more cards than John and Lisa combined. Li Lei has 7 more cards than John. How many cards does Tina have in total?", 
                    "Process quantity": "John and Lisa have a total of x2 cards."}
                    Output: {"explain": "Based on the known information, to know the number of cards Tina has, we must first know how many cards John and Lisa have in total. So the information in 'Process quantity' is necessary", "result": True}
                </Examples>

                Please determine whether it is necessary to solve the process quantity in "Process quantity" in the process of solving the mathematical problem in "question".
                <Input>{input}</Input>
                Please return a dictionary directly as the examples, without any extra text.
                """
        question = " ".join(known_information) + goal
        Input = {"question": question, "Process quantity": sub_target}
        return Dependency_detection_prompt.replace('{input}', f"{Input}")

    def inference_prompt(self, known_information: List[str], additional_known_information: List[str],
                         goal: str) -> str:
        Inference_prompt = """
                <Instruction>
                    The input is a math problem. Please fully consider the known information in the question and solve the problem with the least amount of calculation.
                    
                    Finally, please output a dictionary with two keys "explain" and "result".
                    The "explain" key contains a string that explains the derivation process of the result. 
                    The "result" key contains the final answer to the question.
                </Instruction>

                <Examples>                                                            
                    Input: Tina has 11 pencils. Peter has 6 pencils. Lisa has twice as many pencils as Tina and Peter combined. John has one less pencil than Lisa. Lisa has 34 pencils in total. How many pencils does John have in total? 
                    Output:{"explain": "John has one less pencil than Lisa. Lisa has 34 pencils in total. So John has (34 - 1 = 33) pencils.", "result": "John has 33 pencils in total."}
                
                    Input: Tina has 11 pencils. Peter has 6 pencils. Lisa has twice as many pencils as Tina and Peter combined. John has one less pencil than Lisa. How many pencils does John have in total? 
                    Output:{"explain": "John has one less pencil than Lisa. Lisa has twice as many pencils as Tina and Peter combined. So John has ((11 + 6) * 2 - 1 = 33) pencils.", "result": "John has 33 pencils in total."}
                </Examples>

                Please answer the following math questions:
                <Input>{input}</Input>
                Please return a dictionary directly as the examples, without any extra text.
                """
        Known_information = known_information + additional_known_information
        Input = " ".join(Known_information) + goal
        return Inference_prompt.replace('{input}', f"{Input}")

    def summarization_prompt(self, final_prompt: Dict) -> str:
        Summarization_prompt = """
                <Instruction>
                    The input is a dictionary type data. 
                    The "question" key stores a math problem.
                    The "Reasoning steps" key stores the reasoning steps for the question in "question".
                    
                    Please solve the math problem in "question" according to the reasoning steps provided in "Reasoning steps".
                    
                    Returns a dictionary containing two keys, "explain" and "result". 
                    The "explain" key stores a string that explains the result base on each step.
                    The "result" key stores an number, it is the answer of the question. 
                </Instruction>
                
                <Examples>
                    Input: {"question": "There are 5 basketballs in box A. There are 6 basketballs in box B. The number of basketballs in box C is twice the number of basketballs in box A and box B combined. The number of basketballs in box D is 3 more than in box C. How many basketballs are there in box D?", 
                    "Reasoning steps": ["There are 5 basketballs in box A and 6 basketballs in box B. So there are (5+6=11) basketballs in box A and box B.", "The number of basketballs in box C is twice the number of boxes A and B, so there are (11*2=22) basketballs in box C.", "There are 3 more basketballs in box D than in box C, so there are (22+3=25) basketballs in box D."]}
                    Output: {"explain": "There are (5+6=11) basketballs in frame A and frame B. There are (11*2=22) basketballs in box C. There are (22+3=25) basketballs in box D. So the answer to the question is 25", "result": 25}
                </Examples>

                Please solve the math problem in "question" according to the reasoning steps provided in "Reasoning steps".
                <Input>{input}</Input>
                Please return a dictionary directly as the examples, without any extra text.
                """
        print("summarization_prompt: ", final_prompt)
        return Summarization_prompt.replace('{input}', f"{final_prompt}")

    def Answer_to_known_prompt(self, pre_result: list[str], additional_known_conditions: list[str]) -> str:
        answer_to_known_prompt = """
                <Instruction>
                    The input is a dictionary with two keys, "known information" and "information". 
                    The "known information" key stores some known information. 
                    The "information" key also stores some strings, each containing some symbolic quantity such as "x1".
                    
                    Please replace the symbolic quantity in each string in the key "information", such as x1, with the exact number according to the information provided in the key "known information".
                    
                    Finally, please output a string list. Each string in the list is the replacement result.
                </Instruction>

                <Examples>                                                            
                    Input: {"pre result": ["There are 3 people who have 3 sodas. The people who have 4 sodas have 8 sodas in total."], 
                    "additional known conditions": ["There are x2 people have 3 sodas.", "The person who owns 4 sodas owns x4 sodas in total."]}
                    Output: ["There are 3 people have 3 sodas.", "The person who owns 4 sodas owns 8 sodas in total."]
                </Examples>

                Please replace the corresponding symbolic quantity in the key "information" with the corresponding value of the key "known information" in the following dictionary
                <Input>{input}</Input>
                Please return a string array directly as the examples, without any extra text.
                """
        pre_result = " ".join(pre_result)
        Input = {"known information": pre_result, "information": additional_known_conditions}
        print("Answer_to_known_prompt: ", Input)
        return answer_to_known_prompt.replace('{input}', f"{Input}")
