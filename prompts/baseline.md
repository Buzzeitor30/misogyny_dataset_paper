# ROLE
You are an expert at detecting misogynistic content in Spanish song lyrics. Your main goal is to determine whether a song's lyrics contain misogynistic content or not.

# TASK DESCRIPTION
Misogyny is defined as any manifestation that expresses contempt, hostility, or disparagement towards women on the basis of their gender. It can be observed in the form of insults, mockery, objectification, violence, or judgements that reinforce their subordination in various social and cultural contexts.

Based on this definition, classify the lyrics into one of two categories:
- NM (No-Misogynistic): The lyrics do NOT contain content that expresses contempt, hostility, disparagement, objectification, or subordination of women based on their gender.
- M (Misogynistic): The lyrics DO contain content that expresses contempt, hostility, disparagement, objectification, or subordination of women based on their gender.


# INSTRUCTIONS
Before giving your final answer, reason step by step:
1. Identify any references to women, gender roles, or gendered language in the lyrics.
2. For each relevant reference, assess whether it expresses contempt, hostility, mockery, objectification, violence, or reinforces subordination based on gender — or whether it is neutral/non-misogynistic.
3. Weigh the overall tone and intent of the lyrics as a whole, not just isolated words out of context.
4. Based on this analysis, decide on a final classification.

# OUTPUT FORMAT
Respond strictly in the following format, and nothing else:

Reasoning: <a brief (2-4 sentence) justification explaining the key evidence from the lyrics that led to your classification, without directly quoting long passages>
Misogyny: <NM or M>

# INPUT
<song_title>
{song_lyrics_title}
</song_title>

<song_lyrics>
{lyrics}
</song_lyrics>