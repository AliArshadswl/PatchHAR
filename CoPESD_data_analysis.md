# 📊 ANALYZING SAMPLE DATA

---

## SAMPLE DATA ANALYSIS

**Sample size:** 4 samples

### Main tags in sample
| Tag | Samples |
|------|----------|
| single_phrase | 1 |
| grounding | 1 |
| detailed_description | 1 |
| visual_QA | 1 |

### Sub tags in sample
| Sub-tag | Samples |
|----------|----------|
| Knife_action_performer | 1 |
| grounding | 1 |
| detailed_description | 1 |
| surgical_instrument_action | 1 |

### Question patterns
| Pattern | Count |
|----------|--------|
| what_questions | 1 |
| where_questions | 1 |
| give_questions | 1 |
| can_you_questions | 1 |

### Answer characteristics
| Type | Count |
|-------|--------|
| Bounding boxes | 2 |
| Single words | 1 |
| Long descriptions | 2 |

---

## 🎯 KEY INSIGHTS FROM SAMPLE
- Multi-task dataset with 4 main task types  
- Mix of short answers (single phrases) and complex descriptions  
- Spatial reasoning with bounding box coordinates  
- Surgical domain specialization  
- Requires understanding of surgical instruments and procedures  

---

# 🧠 CoPESD DATASET ANALYSIS FOR VLM FINE-TUNING

✅ **Successfully loaded 248,556 samples from dataset**

---

## 1. BASIC DATASET STATISTICS

| Metric | Value |
|---------|--------|
| Total samples | 248,556 |
| Unique images | 15,674 |
| Samples per image | 15.86 |

---

## 2. MAIN TAG DISTRIBUTION

**Number of unique main tags:** 5

| Main Tag | Samples | Percentage |
|-----------|----------|-------------|
| single_phrase | 62,696 | 25.2% |
| region_based | 62,696 | 25.2% |
| visual_QA | 62,467 | 25.1% |
| grounding | 45,023 | 18.1% |
| detailed_description | 15,674 | 6.3% |

---

## 3. SUB-TAG DISTRIBUTION

**Number of unique sub-tags:** 14

**Top 20 sub-tags**

| Sub-tag | Count | % |
|----------|--------|---|
| grounding | 45,023 | 18.1% |
| Knife_action_performer | 19,870 | 8.0% |
| Knife_action | 19,682 | 7.9% |
| Forceps_action | 17,478 | 7.0% |
| Forceps_action_performer | 17,329 | 7.0% |
| Forceps_action_target | 16,820 | 6.8% |
| Knife_action_target | 16,421 | 6.6% |
| Knife_action_direction | 16,126 | 6.5% |
| surgical_instrument_number | 15,674 | 6.3% |
| surgical_instrument | 15,674 | 6.3% |
| surgical_instrument_action | 15,674 | 6.3% |
| detailed_description | 15,674 | 6.3% |
| surgical_instrument_position | 15,445 | 6.2% |
| Forceps_action_direction | 1,666 | 0.7% |

---

## 4. MAIN TAG → SUB-TAG MAPPING

### detailed_description (1 sub-tag)
→ detailed_description : 15,674 samples  

### grounding (1 sub-tag)
→ grounding : 45,023 samples  

### region_based (8 sub-tags)
→ Forceps_action : 8,792  
→ Forceps_action_direction : 846  
→ Forceps_action_performer : 8,652  
→ Forceps_action_target : 8,375  
→ Knife_action : 9,894  
→ Knife_action_direction : 8,012  
→ Knife_action_performer : 9,912  
→ Knife_action_target : 8,213  

### single_phrase (8 sub-tags)
→ Forceps_action : 8,686  
→ Forceps_action_direction : 820  
→ Forceps_action_performer : 8,677  
→ Forceps_action_target : 8,445  
→ Knife_action : 9,788  
→ Knife_action_direction : 8,114  
→ Knife_action_performer : 9,958  
→ Knife_action_target : 8,208  

### visual_QA (4 sub-tags)
→ surgical_instrument : 15,674  
→ surgical_instrument_action : 15,674  
→ surgical_instrument_number : 15,674  
→ surgical_instrument_position : 15,445  

---

## 5. QUESTION ANALYSIS

**Question statistics**
| Metric | Value |
|---------|--------|
| Average length | 13.8 words |
| Max length | 26 words |
| Min length | 5 words |

**Question patterns**
| Pattern | Count | % |
|----------|--------|--|
| what_is | 97,898 | 39.4% |
| other_patterns | 43,143 | 17.4% |
| can_you_questions | 36,714 | 14.8% |
| where_questions | 32,853 | 13.2% |
| what_other | 24,838 | 10.0% |
| what_are | 5,886 | 2.4% |
| which_questions | 4,942 | 2.0% |
| give_provide_questions | 2,282 | 0.9% |

---

## 6. ANSWER ANALYSIS

**Answer statistics**
| Metric | Value |
|---------|--------|
| Average length | 21.0 words |
| Max length | 510 words |
| Min length | 0 words |

**Answer types**
| Type | Count | % |
|-------|--------|--|
| single_token | 38,541 | 15.5% |
| long_description | 126,158 | 50.8% |
| short_phrase | 23,408 | 9.4% |
| bounding_box | 45,023 | 18.1% |
| numbered_list | 15,426 | 6.2% |

---

## 7. IMAGE PATH ANALYSIS

**Image statistics**
| Metric | Value |
|---------|--------|
| Unique image paths | 15,674 |

**Session distribution**
| Session | Samples | % |
|----------|----------|--|
| 232521 | 37,495 | 15.1% |
| 003054 | 30,908 | 12.4% |
| 000937 | 30,411 | 12.2% |
| 012626 | 25,247 | 10.2% |
| 001841 | 23,840 | 9.6% |
| 010300 | 20,893 | 8.4% |
| 020502 | 20,872 | 8.4% |
| 001815 | 18,183 | 7.3% |
| unknown | 17,836 | 7.2% |
| 014014 | 15,592 | 6.3% |
| 222032 | 5,535 | 2.2% |
| 012710 | 1,744 | 0.7% |

---

## 8. TASK-SPECIFIC ANALYSIS

### SINGLE_PHRASE Task
- Samples: 62,696  
- Avg question length: 16.5 words  
- Avg answer length: 1.7 words  
- Common sub-tags: Knife_action_performer, Knife_action, Forceps_action, Forceps_action_performer, Forceps_action_target  

### REGION_BASED Task
- Samples: 62,696  
- Avg question length: 13.5 words  
- Avg answer length: 11.4 words  
- Common sub-tags: Knife_action_performer, Knife_action, Forceps_action, Forceps_action_performer, Forceps_action_target  

### VISUAL_QA Task
- Samples: 62,467  
- Avg question length: 10.5 words  
- Avg answer length: 19.3 words  
- Common sub-tags: surgical_instrument_number, surgical_instrument, surgical_instrument_action, surgical_instrument_position  

### GROUNDING Task
- Samples: 45,023  
- Avg question length: 17.1 words  
- Avg answer length: 4.0 words  
- Common sub-tags: grounding  

### DETAILED_DESCRIPTION Task
- Samples: 15,674  
- Avg question length: 8.1 words  
- Avg answer length: 191.7 words  
- Common sub-tags: detailed_description  
