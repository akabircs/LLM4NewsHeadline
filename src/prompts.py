\
from __future__ import annotations

from typing import Mapping, Sequence


def _belin_instruction(language: str, variant: str) -> str:
    if language == "xlp" and variant == "baseline":
        return "### Instruction: Generate a news headline in Bengali for the given news article."
    if language == "bnap" and variant == "baseline":
        return "### নির্দেশনা: প্রদত্ত সংবাদ প্রতিবেদনের জন্য একটি বাংলা শিরোনাম তৈরি করুন।"
    if language == "xlp" and variant == "multigen":
        return (
            "### Instruction: Generate a news headline for the given news article considering the sentiment of the news, "
            "the category of the news, and the perspective of the news. Detailed information on these inputs is given below.\n\n"
            "News Sentiment: The sentiment of the article can be positive, negative, or neutral. A positive sentiment indicates "
            "support, a negative sentiment conveys criticism, and a neutral sentiment maintains impartiality.\n"
            "News Category: The religious affiliation of the article, which can be Islam, Hinduism, Christianity, Buddhism, or Others.\n"
            "News Aspect: The primary focus or theme of the article, such as religious reports, festivals, education, or culture."
        )
    if language == "bnap" and variant == "multigen":
        return (
            "### নির্দেশনা: প্রদত্ত সংবাদ প্রতিবেদনের জন্য সংবাদের অনুভূতি, সংবাদের বিভাগ এবং সংবাদের দৃষ্টিভঙ্গির ভিত্তিতে "
            "একটি বাংলা শিরোনাম তৈরি করুন। এই ইনপুটগুলির বিস্তারিত তথ্য নীচে দেওয়া হল।\n\n"
            "সংবাদের অনুভূতি: সংবাদের অনুভূতি ইতিবাচক, নেতিবাচক বা নিরপেক্ষ হতে পারে। একটি ইতিবাচক অনুভূতি সমর্থন নির্দেশ করে, "
            "একটি নেতিবাচক অনুভূতি সমালোচনা প্রকাশ করে এবং একটি নিরপেক্ষ অনুভূতি নিরপেক্ষতা বজায় রাখে।\n"
            "সংবাদের বিভাগ: এটি সংবাদের ধর্মীয় সংশ্লিষ্টতা নির্দেশ করে, যা ইসলাম, হিন্দুধর্ম, খ্রিস্টধর্ম, বৌদ্ধধর্ম বা অন্যান্য হতে পারে।\n"
            "সংবাদের দৃষ্টিভঙ্গি: এটি হলো সংবাদের প্রাথমিক ফোকাস বা থিম, যেমন ধর্মীয় প্রতিবেদন, উৎসব, শিক্ষা বা সংস্কৃতি।"
        )
    raise ValueError(f"Unsupported language/variant: {language}/{variant}")


def _belin_block(language: str, variant: str, item: Mapping[str, str], headline: str | None = None, example: bool = False) -> str:
    prefix = "# Example:\n" if example and language == "xlp" else ("# উদাহরণ:\n" if example else "")
    if language == "xlp":
        if variant == "baseline":
            block = f"# News Article: {item['article']}\n"
        else:
            block = (
                f"# News Sentiment: {item['sentiment']}\n"
                f"# News Category: {item['category']}\n"
                f"# News Aspect: {item['aspect']}\n"
                f"# News Article: {item['article']}\n"
            )
        block += "# News Headline: " + (headline or "")
    else:
        if variant == "baseline":
            block = f"# সংবাদ প্রতিবেদন: {item['article']}\n"
        else:
            block = (
                f"# সংবাদের অনুভূতি: {item['sentiment']}\n"
                f"# সংবাদের বিভাগ: {item['category']}\n"
                f"# সংবাদের দৃষ্টিভঙ্গি: {item['aspect']}\n"
                f"# সংবাদ প্রতিবেদন: {item['article']}\n"
            )
        block += "# সংবাদের শিরোনাম: " + (headline or "")
    return prefix + block


def build_belin_prompt(
    *, language: str, variant: str, article: str, sentiment: str = "", category: str = "", aspect: str = "",
    demonstrations: Sequence[Mapping[str, str]] = (), shots: int = 0,
) -> str:
    if language not in {"xlp", "bnap"} or variant not in {"baseline", "multigen"}:
        raise ValueError("language must be xlp/bnap and variant must be baseline/multigen")
    if not 0 <= shots <= 3:
        raise ValueError("shots must be 0, 1, 2, or 3")
    if shots > len(demonstrations):
        raise ValueError("Not enough demonstrations were supplied")

    parts = [_belin_instruction(language, variant), ""]
    for demo in demonstrations[:shots]:
        parts.append(_belin_block(language, variant, demo, headline=demo["headline"], example=True))
        parts.append("")
    target = {"article": article, "sentiment": sentiment, "category": category, "aspect": aspect}
    parts.append(_belin_block(language, variant, target, headline=None, example=False))
    return "\n".join(parts)


def build_shironaam_xlp_multigen_prompt(
    *, article: str, category: str, tags: str, image_caption: str,
    demonstrations: Sequence[Mapping[str, str]], shots: int = 3,
) -> str:
    if not 0 <= shots <= 3:
        raise ValueError("shots must be 0, 1, 2, or 3")
    instruction = (
        "### Instruction: Generate a news headline for the given news article considering the category, topic words and image caption "
        "of the news. Detailed information on these inputs is given below.\n\n"
        "News Category: The broad section or domain of the news (e.g., Politics, Sports, Religion, Technology).\n"
        "News Topic words: Important keywords or topic words that highlight the main subjects, entities, or themes discussed in the article.\n"
        "News Image Caption: A brief description of the image associated with the news article."
    )
    parts = [instruction, ""]
    for i, demo in enumerate(demonstrations[:shots], start=1):
        parts.append(
            f"# Example-{i}:\n"
            f"# News Category: {demo['category']}\n"
            f"# News Topic words: {demo['tags']}\n"
            f"# News Image Caption: {demo['image_caption']}\n"
            f"# News Article: {demo['article']}\n"
            f"# News Headline: {demo['headline']}"
        )
        parts.append("")
    parts.append(
        f"# News Category: {category}\n"
        f"# News Topic words: {tags}\n"
        f"# News Image Caption: {image_caption}\n"
        f"# News Article: {article}\n"
        "# News Headline: "
    )
    return "\n".join(parts)
