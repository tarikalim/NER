import json


def fix_ner_tags_v2(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed_data = []

    # Data bir listeyse doğrudan işlem yap
    for item in data:
        if isinstance(item, dict):  # Eğer eleman bir sözlükse
            paragraphs = item.get("paragraphs", [])
            for paragraph in paragraphs:
                tokens = []
                ner_tags = []
                for entity in paragraph:
                    for key, value in entity.items():
                        words = value.split()  # Çoklu kelimeler için ayrıştırma
                        if key == "PERSON":
                            tags = ["B-PER"] + ["I-PER"] * (len(words) - 1)
                        elif key == "GPE":
                            tags = ["B-LOC"] + ["I-LOC"] * (len(words) - 1)
                        elif key == "ORG":
                            tags = ["B-ORG"] + ["I-ORG"] * (len(words) - 1)
                        elif key == "TIME":
                            tags = ["B-TIME"] + ["I-TIME"] * (len(words) - 1)
                        elif key == "MISC":
                            tags = ["B-MISC"] + ["I-MISC"] * (len(words) - 1)
                        else:
                            tags = ["O"] * len(words)

                        tokens.extend(words)
                        ner_tags.extend(tags)

                fixed_data.append({"tokens": tokens, "ner_tags": ner_tags})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=4)


# Kullanım
fix_ner_tags_v2("einstein_cot.json", "fixed_einstein_cot.json")
