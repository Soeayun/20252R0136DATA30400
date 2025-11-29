# [cite_start]DATA304 Big Data Analysis: Tutorial Class #3 [cite: 1, 2]

[cite_start]**SeongKu Kang** **Korea University** [https://www.idea.korea.ac.kr/](https://www.idea.korea.ac.kr/) [cite: 3, 4, 5]

---

## [cite_start]Final Project: Hierarchical Multi-Label Text Classification [cite: 13, 14]

[cite_start]**Key Reference:** **TaxoClass: Hierarchical Multi-Label Text Classification Using Only Class Names** [cite: 7]  
[cite_start]*Jiaming Shen, Wenda Qiu, Yu Meng, Jingbo Shang, Xiang Ren, Jiawei Han* University of Illinois at Urbana-Champaign, University of California, San Diego, University of Southern California [cite: 8, 9]  
[cite_start]*Emails:* {js2, qiuwenda, yumeng5, hanj}@illinois.edu, jshang@ucsd.edu, xiangren@usc.edu [cite: 10, 11]

> [cite_start]You are encouraged to use this paper as a key reference of this project. [cite: 12]

---

## [cite_start]Task Description [cite: 17]

[cite_start]**Task:** Hierarchical Multi-Label Text Classification [cite: 17]
* [cite_start]Your task is to perform product review classification **without using any labeled data**. [cite: 18]
* [cite_start]Each review is associated with multiple product categories that are organized in a hierarchical taxonomy. [cite: 19]

### Example Case

[cite_start]**Document:** "When our son was about 4 months old, our doctor said we could give him crafted cereal. We bought this product and put it in his bottle. He loved this stuff! This cereal digests well and didn't lock up his bowels at all. We highly recommend this cereal." [cite: 20, 21, 22]

[cite_start]**Class Taxonomy (Hierarchy):** [cite: 23]
* [cite_start]**Root** [cite: 24]
    * [cite_start]**baby product** [cite: 25]
        * [cite_start]diapering [cite: 26]
        * [cite_start]nursery [cite: 29]
        * [cite_start]**feeding** [cite: 27]
            * [cite_start]**baby food** [cite: 28]
                * [cite_start]beverages [cite: 30]
                * [cite_start]baby formula [cite: 31]
                * [cite_start]**crafted cereal** [cite: 31]
                * [cite_start]**baby cereal** [cite: 33]
    * [cite_start]**grocery & gourmet food** [cite: 28]
        * ...
        * [cite_start]toddler fruit [cite: 32]

**Analysis of the Example:**
* [cite_start]The document (review) is related to a total of **five classes**: `baby product`, `feeding`, `crafted cereal`, `baby food`, `baby cereal`. [cite: 37]
* [cite_start]"grocery & gourmet food" is **not** an answer since the review is about a specific product (crafted cereal), not the broader grocery category. [cite: 38]

### Dataset Statistics
[cite_start]You are given: [cite: 34]
* [cite_start]**29,487 reviews (training)** [cite: 36]
* [cite_start]**19,658 reviews (test)** [cite: 36]
* [cite_start]**531 product classes and their hierarchy** [cite: 36]

---

## New Challenges

### [cite_start]1. Multiple labels for each document [cite: 42]
[cite_start]In the given dataset, each document is associated with **at least two and at most three labels**. [cite: 43]

> [cite_start]**Question:** How can we determine which classes are most strongly associated with a given document? [cite: 57]

### [cite_start]2. Hierarchy of classes [cite: 61]
[cite_start]Unlike previous assignments (PA#3) where all classes were flat, here they are organized in a **hierarchical taxonomy**. [cite: 61]
* [cite_start]Each node represents a category, and child nodes inherit semantic relationships from their parents. [cite: 62]
* [cite_start]When generating silver labels, constructing label embeddings, or applying pseudo-labeling, these hierarchical relationships need to be reflected to ensure consistency across related classes. [cite: 73]

---

## Utilizing Large Language Models (LLMs)

**Good news! [cite_start]We can use Large Language Models (LLMs)** [cite: 76]
* [cite_start]LLMs have strong language understanding and can help address the scarcity of labels. [cite: 77]
* [cite_start]One intuitive approach is to simply ask the LLM to find the most relevant classes for each document. [cite: 78]

**Example Prompting:**
> [cite_start]**Prompt:** "find relevant classes for this document" [cite: 82]  
> [cite_start]**LLM Output:** "Crafted cereal, baby food..." [cite: 95]

### [cite_start]LLMs empower you only with proper usage [cite: 97]
* [cite_start]LLMs are **not** a free, all-powerful solution. [cite: 98]
* [cite_start]You may spend a lot of money and time, yet achieve limited performance if used naively. [cite: 99]

[cite_start]**Performance and Cost Comparison Table:** [cite: 100]

| Methods | Dataset 1 (Example-F1) | Dataset 1 (P@1 / P@3) | Dataset 1 (Est. Cost) | Dataset 1 (Est. Time) | Dataset 2 (Example-F1) | Dataset 2 (P@1 / P@3) | Dataset 2 (Est. Cost) | Dataset 2 (Est. Time) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-3.5-turbo** | 0.5164 | 0.6807 / 0.4752 | $60 | 240 mins | 0.4816 | 0.5328 / 0.4547 | $80 | 400 mins |
| **GPT-3.5-turbo (level)** | 0.6621 | 0.8574 / 0.6444 | $20 | 800 mins | 0.6649 | 0.8301 / 0.6488 | $60 | 1,000 mins |
| **GPT-4+** | 0.6994 | 0.8220 / 0.6890 | $800 | 400 mins | 0.6054 | 0.6520 / 0.5920 | $2,500 | 1,000 mins |
| **BERT-based classifier** | 0.6483 | 0.6421 / 0.8505 | <$1 | 3 mins | 0.8633 | 0.9351 / 0.8633 | <$1 | 7 mins |

*Note: BERT-based classifier is trained with the help of LLMs. [cite_start]Costs and times shown above refer to inference only.* [cite: 101]

### [cite_start]Why limited? [cite: 102]
* [cite_start]There are too many classes, and hierarchical relations exist among them. [cite: 103]
* [cite_start]When the prompt becomes too long, the LLM cannot process it efficiently. [cite: 104]
* The information for each class is very limited. [cite_start]We only have the **category name**. [cite: 105]

---

## Strategies to Improve Performance

### [cite_start]1. Generate more contexts for each class [cite: 109]
* [cite_start]Class information is often too limited (only category names). [cite: 110]
* [cite_start]We can enrich class information by generating additional contexts (e.g., key terms). [cite: 111]

[cite_start]**Example Instruction:** [cite: 115]
> `[Target Class]` is a product class in Amazon and is the subclass of `[Parent Class]`.  
> Please generate 10 additional key terms about the `[Target Class]` that are relevant to `[Target Class]` but irrelevant to `[Sibling Classes]`.  
> [cite_start]Please split the additional key terms using commas. [cite: 116, 117, 118]

**Result for target class "crafted cereal":**
> [cite_start]multigrain cereal, iron-fortified cereal, stage 1 baby food, ... [cite: 129, 130]

### [cite_start]2. Simplify the problem of LLMs [cite: 134]
* [cite_start]First, narrow down the most probable candidate classes using reasonable heuristics or models. [cite: 135]
* [cite_start]Then, ask the LLM to identify the most relevant ones among the **reduced set**. [cite: 136]

[cite_start]**Example Instruction:** [cite: 140]
> You will be provided with an Amazon product review, and select its product types from the following candidates:  
> [cite_start]`baby product, feeding, nursery, crafted cereal, ...` [cite: 141, 142]

**Example Output:**
> [cite_start]feeding, crafted cereal [cite: 154]

### [cite_start]3. Focus on the parts where your classifier struggles [cite: 157]
* [cite_start]For samples that your classifier already handles well (relatively easy data), the help of LLMs may not be necessary. [cite: 158]
* [cite_start]Instead, leverage LLMs **selectively** for the parts where your model struggles, using them to provide additional or corrective information. [cite: 159]

*And, there are many more possible directions! [cite_start]This remains an active and evolving research area.* [cite: 160, 161]

---

## [cite_start]Task Summary & Resources [cite: 164]

### [cite_start]Provided Resources [cite: 165]
1.  [cite_start]**Product review data:** 29,487 reviews (training), 19,658 reviews (test) [cite: 166]
2.  [cite_start]**Classes:** 531 product categories and their hierarchy [cite: 167]
3.  [cite_start]**Class-related keywords:** 10 keywords per class [cite: 168]
    * [cite_start]These are generated using GPTs with instructions to generate terms relevant to the target class but irrelevant to sibling classes. [cite: 169, 170, 171]

### [cite_start]API Usage Limit [cite: 173]
* [cite_start]**You can additionally use 1,000 API calls of LLMs.** [cite: 173]
* [cite_start]You may use the GPT-4o mini API (approximately $1) or any other freely available LLM. [cite: 174]

---

## [cite_start]General Instructions: Submission & Grading [cite: 177]

### [cite_start]How to Submit [cite: 178]
1.  [cite_start]**Prediction Results (Kaggle):** Submit your prediction results on Kaggle. [cite: 179, 180]
2.  **Code (GitHub):** Submit your code via GitHub. [cite_start]The repository must include all components to reproduce results. [cite: 181, 182]
3.  [cite_start]**Report & GitHub link (LMS):** The report should be written in English, up to 8 pages. [cite: 183, 184]

### [cite_start]Grading Criteria [cite: 185]
1.  [cite_start]**Performance (50%):** [cite: 187]
    * [cite_start]Your performance on Kaggle will be evaluated (Private leaderboard used). [cite: 188]
    * [cite_start]To get credit, your results must be reproducible. [cite: 189]
2.  [cite_start]**Report Quality (50%):** [cite: 190]
    * [cite_start]Evaluation based on completeness, clarity, and organization. [cite: 191]

---

## [cite_start]General Instructions: Extra Credits [cite: 194]

[cite_start]**Extra Credits (+10%):** Specifically requested by the College of Informatics. [cite: 195]
1.  [cite_start]**At least 10 GitHub commits** (5%). [cite: 196]
2.  [cite_start]**At least 90% utilization of the allocated AWS resources** (5%). [cite: 197]
    * [cite_start]Information about remaining AWS usage time will be provided periodically via LMS. [cite: 199]
    * [cite_start]Full credit will be given as long as the above criteria are met. [cite: 200]

### [cite_start]GitHub Details [cite: 201]
* [cite_start]Repositories will be collected and analyzed as evidence of open-source contributions for university-level policy decisions. [cite: 202, 203]
* [cite_start]**You must make at least 10 commits.** [cite: 204]
* [cite_start]**Naming Format:** `https://github.com/Your GitHub ID/20252R0136DATA30400` [cite: 206]
* [cite_start]Recommendation: Keep repository private before the deadline, then make it public after submission. [cite: 207]

---

## [cite_start]General Instructions: Honor Code [cite: 209]

1.  [cite_start]All submitted code and report must be your **own work**. [cite: 210]
2.  [cite_start]The use of any data other than the provided dataset is **strictly prohibited**. [cite: 212]
    * [cite_start]If you use public pretrained models (other than BERT), make sure they are **not fine-tuned on Amazon data**. [cite: 213]
3.  [cite_start]You may use LLMs to improve the quality of your report, but you must take full responsibility for all content. [cite: 214, 215]
4.  [cite_start]**LLM Usage:** If you used an LLM for the task, the prompt and output file (including results up to 1,000 calls) **must be provided**. [cite: 216]
    * [cite_start]Exceeding the specified number of API calls will be considered inappropriate behavior. [cite: 217]
    * [cite_start]We may request supporting materials (e.g., OpenAI dashboard logs) to verify LLM usage. [cite: 218]

---

## [cite_start]General Instructions: Reproducibility [cite: 222]

* [cite_start]You are responsible for ensuring your results are fully reproducible. [cite: 223]
* The submitted code must be executable **without any modification**. [cite_start]We will clone and run your code. [cite: 224, 225]
* [cite_start]**Submissions that cannot be reproduced (within a reasonable range) will receive no credit.** [cite: 227]
* [cite_start]Provide a clear/detailed description in `README.md`. [cite: 228]
* [cite_start]If multiple steps are involved, include a **shell script (.sh)** to run the entire process. [cite: 229]
* [cite_start]If using large intermediate files/models, store them externally (e.g., Google Drive) and provide the download link in `README.md`. [cite: 230]
* [cite_start]*Note: The instructor is very familiar with the dataset and can identify irregular behavior easily.* [cite: 231, 232]

---

## [cite_start]General Instructions: Submission Format [cite: 235]

* [cite_start]**Kaggle team name:** `(AWS account ID)` [cite: 237]
* [cite_start]**GitHub repository name:** Follow the format mentioned above. [cite: 238]
* [cite_start]Do not risk losing credit due to incorrect naming or submission format. [cite: 239]