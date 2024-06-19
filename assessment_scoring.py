import pandas as pd
import os
import openpyxl
#from nltk import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import analysis_modules
import nltk

nltk.download('punkt')
nltk.download('stopwords')
path = os.getenv('FILE_PATH')
#os.chdir(path)

#%%
# Load Data
file_name = 'input/response.xlsx'  # non_priority_cleaned.csv or response.xlsx
df = analysis_modules.response_load(file_name)

results = []
similarity_scores = []
domain_concat = []
scoring = []
#%%
for _, row in df.iterrows():
    domain = row['Domain']
    capability = row["Capability"]
    level = row['Level']
    question = row['Question']
    answers = row['Answers']
    criteria1 = row['Capability-1']
    criteria2 = row['Capability-2']
    # ORIGINAL PROMPT: prompt = f"""Given the criteria and text provided below, evaluate the alignment between the two. Please analyze the alignment
    #between the following criteria and text. Summarize the alignment strength as either weak, moderate, or strong, and
    #provide a brief justification for this assessment in one or two sentences. The response should naturally incorporate
    #both the evaluation of alignment strength and the reasoning behind it.""
    prompt = f"""Analyze the following text and compare it to the given criteria. Gauge the alignment of the text to the 
    criteria and output an alignment of either weak, moderate, or strong. Additionally, include 3 short bullet 
    points as a justification for the alignment rating. These bullet points should be concise and fact-based statements.\n\n
    Text: {answers}\n\n
    Criteria: {criteria1}\n\n
    Output format:\n\n
    Alignment: [Weak/Moderate/Strong]\n\n
    -Bullet point 1\n
    -Bullet point 2\n
    -Bullet point 3\n"""

    alignment = analysis_modules.ia_analysis(prompt)
    alignment = analysis_modules.clean_and_normalize_text(alignment)
    alignment = analysis_modules.to_sentence_case(alignment)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    similarity = analysis_modules.compute_cosine_similarity(criteria1, answers, tokenizer, model)
    similarity = round(similarity, 2)
    similarity_scores.append(similarity)


    maturity_score = analysis_modules.get_maturity_score(alignment)


    # DevelopOpportunities and Recommendations
    #ORIGINAL PROMPT: prompt = f"""PLEASE FOLLOW THESE INSTRUCTIONS CAREFULLY AND PRECISELY: Conduct an analysis based on the provided
    #    Level 1 maturity assessment and the defined criteria for Level 2 maturity. Then, articulate a clear and concise
    #    narrative in one or two sentences. This narrative should implicitly identify a key area where improvement is needed
    #    without directly mentioning it as an 'opportunity for improvement' or using any similar labels. The response should
    #    seamlessly integrate the identified area into the narrative, focusing solely on the description of the area needing
    #    improvement without any explicit labeling or recommendations. Remember, do not use any form of the phrase
    #    'opportunity for improvement' or include any additional formatting."
    #    "Level 1 maturity assessment: {assessment}"
    #    "Criteria for Level 2 maturity: {criteria}"
    #    """
    prompt = f"""Conduct an analysis based on the provided Level 1 maturity assessment and the defined criteria for 
            Level 2 maturity. Then, develop 3 summary bullet points, suitable for a powerpoint slide. These bullets 
            should implicitly identify a key area where improvement is needed without directly mentioning it as an 
            'opportunity for improvement' or using any similar labels. focusing solely on the description of the area needing 
            improvement without any explicit labeling or recommendations. Remember, do not use any form of the phrase 
            'opportunity for improvement' or include any additional formatting."
            "Level 1 maturity assessment: {answers}"
            "Criteria for Level 2 maturity: {criteria2}"
            """
    opportunity = analysis_modules.ia_analysis(prompt)
    opportunity = analysis_modules.clean_and_normalize_text(opportunity)
    opportunity = analysis_modules.to_sentence_case(opportunity)

    # ORIGINAL PROMPT: prompt = f"""Your task is to analyze a scenario involving the progression from a Level 1 to Level 2 maturity in a
    #    business context.
    #    Begin by considering a Level 1 Assessment {assessment}.
    #    The criteria for achieving Level 2 maturity: {criteria}.
    #    Given this context, provide an analysis that seamlessly transitions into 1 concise recommendation.
    #    This recommendation should encapsulate a strategy for addressing the noted gaps and aligning with Level 2 maturity criteria.
    #    The insight should be presented in a single paragraph without explicitly stating its purpose as a recommendation or including any form of conclusion.
    #    Key Instructions:
    #    -   Do not explicitly mention the insight is a recommendation.
    #    -	Avoid directly stating the analysis is for transitioning to Level 2 maturity.
    #    -	Provide the insight in one concise paragraph without a concluding statement.
    #    """
    prompt = f"""Your task is to analyze a scenario involving the progression from a Level 1 to Level 2 maturity in a 
            business context. 
            Begin by considering a Level 1 Assessment {answers}. 
            The criteria for achieving Level 2 maturity: {criteria2}. 
            Given this context, provide an analysis that seamlessly transitions into 3 concise bullet point recommendations.
            This recommendation should encapsulate a strategy for addressing the noted gaps and aligning with Level 2 maturity criteria.
            Key Instructions:
            -   Do not explicitly mention the insight is a recommendation.
            -	Avoid directly stating the analysis is for transitioning to Level 2 maturity. 
            -	Provide the insight in one concise paragraph without a concluding statement.
            """
    # Submit to analysis_modules.ia_analysis function to get a new set of recommendations
    recommendation = analysis_modules.ia_analysis(prompt)
    recommendation = analysis_modules.clean_and_normalize_text(recommendation)
    recommendation = analysis_modules.to_sentence_case(recommendation)

    # Develop Backlog, User story and Activities
    prompt = f"""Build an agile backlog for implementation of the following recommendation {recommendation}. 
    The response should only contain the following information: "Backlog Title", "User Story" and a list of 3 - 5 
    implementation activities."""
    backlog = analysis_modules.build_backlog(prompt)

    # Put it All Together
    parts = [
        f"{domain}|{capability}|{level}|{criteria1}|{answers}|{alignment}|{maturity_score}|{similarity}|{opportunity}|{recommendation}|{backlog}"]
    data = [item.split('|') for item in parts]
    results.append(data[0])
    df = pd.DataFrame(results,
                      columns=['Domain', 'Capability', 'Level', 'Criteria', 'Assessment', 'Alignment', 'Maturity Score',
                               'Similarity Score', 'Opportunity', 'Recommendation', 'Backlog'])

    # Build domain summary assessment
    domain_concat.append(pd.DataFrame({"Domain": domain, "Alignment": alignment}, index=[0]))

#%%
# Output the Recommendations
recommendations_output = ['Domain', 'Capability', 'Level', 'Alignment', 'Maturity Score', 'Similarity Score',
                          'Opportunity', 'Recommendation']
# Select these columns and save to CSV
df[recommendations_output].to_csv('output/Recommendations1.csv', index=False)

#%%
# Output the Backlog
backlog_output = ['Domain', 'Capability', 'Level', 'Backlog']
# Select these columns and save to CSV
df[backlog_output].to_csv('output/backlog.csv', index=False)

#%%
# Develop Domain Summary
align_results_df = pd.concat(domain_concat)
align_concat = align_results_df.groupby('Domain')['Alignment'].apply(' '.join).reset_index()
# Save Domain Summary to csv file
align_concat.to_csv('output/priority_domain_summary.csv')
