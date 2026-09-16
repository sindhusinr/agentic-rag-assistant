# Ground-truth test cases used for offline RAG evaluation
EVALUATION_DATASET = [
    {
        "question": "How many Earned Leave days are employees eligible for in a year?",
        "reference_answer": "Employees are eligible for 18 days of Earned Leave per calendar year, credited at 1.5 days per month.",
        "relevant_sections": [
            {"source": "Leave policy.pdf", "section": "1. Earned Leave (EL)"}
        ],
    },
    {
        "question": "Can unused Earned Leave be carried forward?",
        "reference_answer": "Yes. Employees may carry forward unused Earned Leave up to a maximum of one year's entitlement.",
        "relevant_sections": [
            {"source": "Leave policy.pdf", "section": "1. Earned Leave (EL)"}
        ],
    },
    {
        "question": "What happens if an employee takes more Earned Leave than they are entitled to?",
        "reference_answer": "If an employee has taken Earned Leave in excess of their entitlement, they must reimburse the Company for the excess leave at the time of separation.",
        "relevant_sections": [
            {"source": "Leave policy.pdf", "section": "1. Earned Leave (EL)"}
        ],
    },
    {
        "question": "When is a medical certificate required for Sick Leave?",
        "reference_answer": "A medical certificate must be submitted if Sick Leave exceeds three consecutive days.",
        "relevant_sections": [
            {"source": "Leave policy.pdf", "section": "2. Sick Leave (SL)"}
        ],
    },
    {
        "question": "Can unused Sick Leave be carried forward or encashed?",
        "reference_answer": "No. Unused Sick Leave cannot be carried forward to the next year and cannot be encashed.",
        "relevant_sections": [
            {"source": "Leave policy.pdf", "section": "2. Sick Leave (SL)"}
        ],
    },
    {
        "question": "How much paid maternity leave is available?",
        "reference_answer": "Female employees are entitled to 26 weeks of paid maternity leave for up to two children, or 12 weeks if they already have two or more children.",
        "relevant_sections": [
            {"source": "Leave policy.pdf", "section": "3. Parental Leave"}
        ],
    },
    {
        "question": "What parental leave is available to employees who are not entitled to maternity leave?",
        "reference_answer": "Employees who are not entitled to maternity leave and have completed one year of service are eligible for 5 days of fully paid parental leave and 15 working days at 80% base pay.",
        "relevant_sections": [
            {"source": "Leave policy.pdf", "section": "3. Parental Leave"}
        ],
    },
    {
        "question": "Is Leave Without Pay allowed, and whose approval is required?",
        "reference_answer": "Leave Without Pay is generally not allowed, but the Company may grant it for strong and valid reasons on a case-by-case basis. Approval is required from the Immediate Manager, Departmental Manager, and HR Head.",
        "relevant_sections": [
            {"source": "Leave policy.pdf", "section": "4. Leave Without Pay"}
        ],
    },
    {
        "question": "What are the normal working hours for main office employees?",
        "reference_answer": "Main office employees work Monday to Friday from 9:30 a.m. to 6:30 p.m., with a one-hour lunch break.",
        "relevant_sections": [
            {"source": "Employee Handbook.pdf", "section": "15.1 Business Hours"}
        ],
    },
    {
        "question": "How long is the probation period, and can it be extended?",
        "reference_answer": "The probation period is six months. It may be extended by an additional three months if the employee's performance does not meet expectations.",
        "relevant_sections": [
            {"source": "Employee Handbook.pdf", "section": "6. Probationary Period"}
        ],
    },
]