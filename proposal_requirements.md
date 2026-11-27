# AIE231 – Week 9: Project Proposal

**Project:** Bank Marketing Prediction using Neural Networks

يا شباب، المطلوب مننا في Week 9 هو تجهيز الـ Project Proposal، وطبقاً للـ Course Project Specification لازم نخلص 4 مهام أساسية، بالإضافة إلى إنشاء GitHub Repository.

## 1. Problem Definition
تحديد المشكلة اللي المشروع هيحلها بشكل واضح.

**Requirements:**
- Explain the real-world problem.
- Identify who will use the prediction system.
- Define exactly what the model will predict.
- Explain the consequences of incorrect predictions.
- Define the Machine Learning task.

**Our selected problem:**
Predict whether a bank client will subscribe to a term deposit based on their personal, financial, and campaign-related features.

**Acceptance Criteria:**
- [ ] Problem is clearly stated in one paragraph.
- [ ] The target variable is identified.
- [ ] The input features and expected prediction are clear.
- [ ] The real-world importance of the problem is explained.
- [ ] The task is identified as Binary Classification.

## 2. Dataset Selection & Validation

**Selected Dataset:** UCI Bank Marketing Dataset.

**Requirements:**
Verify that the selected dataset satisfies ALL the minimum dataset criteria in the specification.

**Acceptance Criteria:**
- [ ] At least 10,000 rows.
- [ ] At least 10 features.
- [ ] Binary classification with clear class imbalance.
- [ ] Contains real-world imperfections, such as unknown values and mixed feature types.
- [ ] Publicly available with a license permitting academic use.
- [ ] Dataset source and download link are documented.
- [ ] Dataset is confirmed to be suitable for training within the project's compute limitations.

**Dataset information from the specification:**
- 45,211 rows.
- 16 features.
- Binary classification.
- Mixed numerical and categorical features.
- Unknown entries and imbalanced target.

## 3. Primary Evaluation Metric

We must select ONE primary metric and justify why it is suitable for our problem.

**Requirements:**
- Identify the primary evaluation metric.
- Explain why it is more appropriate than relying on accuracy alone.
- Connect the metric to the cost of incorrect predictions.
- Explain the importance of False Positives and False Negatives in our problem.

**Acceptance Criteria:**
- [ ] One primary metric is selected.
- [ ] A clear justification is provided.
- [ ] The justification is related to the business problem.
- [ ] The metric is appropriate for the class distribution and prediction objective.

Note: Do not select a metric without explaining why it fits our problem.

## 4. Team Roles & Responsibilities

**Team size:** 5–6 students.

**Requirements:**
- Assign responsibilities to every team member.
- Define ownership of each work package.
- Ensure that every member has a clear responsibility.

**Acceptance Criteria:**
- [ ] Every team member has an assigned role.
- [ ] Each role has clearly defined responsibilities.
- [ ] No responsibilities are left unassigned.
- [ ] The role distribution is consistent with the project specification.

Suggested roles:
1. Tech Lead.
2. Data Engineer.
3. Baseline & Analysis Engineer.
4. Model Architect.
5. Experimentation Engineer.
6. Evaluation & Deployment Engineer.

For a team of five, roles 1 and 3 are merged as specified in the document.

## 5. GitHub Repository Setup

**Requirements:**
- install git if you don't have in your pc
- git clone https://github.com/3arab-star/Neural-Network-Project/ 
- Create a GitHub repository for the project.
- Add all team members.
- Establish the initial project structure.
- Prepare the repository for collaborative development.

**Acceptance Criteria:**
- [ ] Repository is created and accessible.
- [ ] All team members have access.
- [ ] Project README.md exists.
- [ ] Initial folder structure is prepared.
- [ ] Repository is ready for collaborative development.

---

## Final Week 9 Deliverables

Before considering Week 9 complete, the team must deliver:

1. A documented Problem Statement.
2. A selected and validated Dataset with its source and license.
3. A justified Primary Evaluation Metric.
4. A Team Roles & Responsibilities table.
5. A GitHub Repository link.

**Definition of Done:**

The Week 9 checkpoint is complete when the problem and dataset are chosen, the dataset is verified against the specification criteria, the primary metric is fixed, team roles are assigned, and the GitHub repository is created.

**Important:** Week 9 is about project planning and proposal preparation. EDA, preprocessing, data splitting, and baseline implementation belong to Week 10.
