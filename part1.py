
# PART 1
import transformers
from transformers import pipeline
import transformers
from transformers import pipeline
import torch  # если используешь GPU

pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")

from haystack import Document
from haystack.components.readers import ExtractiveReader

docs = [Document(content="Dearly beloved we gather here to say our goodbyes. Here she lies, no one knew her worth. The late great daughter of Mother Earth. On these nights when we. Celebrate the birth. In that little town of Bethlehem. We raise our glass, you bet your ass to. La vie Boheme. To days of inspiration. Playing hookey, making. Something out of nothing (la vie Boheme). The need to express. To communicate (la vie Boheme). To going against the grain. Going insane, going mad (la vie Boheme). To loving tension, no pension (la vie Boheme). To more than one dimension (la vie Boheme). To starving for attention. Hating convention, hating pretension (la vie Boheme). Not to mention of course. Hating dear old Mom and Dad (la vie Boheme). To riding your bike (la vie Boheme). Midday past the three-piece suits (la vie Boheme). To fruits, to no absolutes. To Absolute, to choice (la vie Boheme). To the Village Voice (la vie Boheme). To any passing fad. To being an us for once, instead of a them. La vie Boheme (la vie Boheme). Hey Mister, she's my sister. So that's five miso soup, four seaweed salad. Three soy burger dinner, two tofu dog platter. And one pasta with meatless balls. Eww. It tastes the same. If you close your eyes. And thirteen orders of fries")]


reader = ExtractiveReader(model="deepset/roberta-base-squad2")
reader.warm_up()

question = "What do they find gross?"
result = reader.run(query=question, documents=docs)

print(result)