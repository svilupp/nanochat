"""
Stage 7: Select top K conversations and convert to NanoChat format.

This script:
1. Loads unique conversations from Stage 6
2. Sorts by quality score
3. Selects top K conversations
4. Converts to NanoChat format (messages only)
5. Saves final dataset
"""

import asyncio

import logfire
from dotenv import load_dotenv

from src.synth_data_pipeline.models import (
    UniqueConversation,
    NanoChatConversation,
    NanoChatMessage,
)
from src.synth_data_pipeline.config import PATHS, FULL_PARAMS
from src.synth_data_pipeline.utils import load_jsonl, save_jsonl

# Load environment variables
load_dotenv()

# Configure logging
logfire.configure(scrubbing=False)


def conversation_to_nanochat(unique_conv: UniqueConversation) -> NanoChatConversation:
    """
    Convert a UniqueConversation to NanoChat format.

    Args:
        unique_conv: UniqueConversation object

    Returns:
        NanoChatConversation (messages only)
    """
    nanochat_messages = [
        NanoChatMessage(role=msg.role, content=msg.content)
        for msg in unique_conv.conversation.messages
    ]

    return NanoChatConversation(messages=nanochat_messages)


async def main(
    input_file: str = None,
    output_file: str = None,
    top_k: int = None,
    min_score: float = None,
):
    """
    Main function to select top K conversations.

    Args:
        input_file: Path to input JSONL file (default from config)
        output_file: Path to output JSONL file (default from config)
        top_k: Number of top conversations to select (default from config)
        min_score: Minimum quality score threshold (default from config)
    """
    # Use defaults from config if not specified
    input_file = input_file or PATHS.stage6_conversations_unique
    output_file = output_file or PATHS.stage7_conversations_final
    top_k = top_k or FULL_PARAMS.top_k
    min_score = min_score or FULL_PARAMS.min_quality_score

    logfire.info(
        "Starting top-K selection",
        input_file=input_file,
        top_k=top_k,
        min_score=min_score,
    )

    # Load unique conversations
    unique_convs = load_jsonl(input_file, model_class=UniqueConversation)
    logfire.info(f"Loaded {len(unique_convs)} unique conversations")

    # Filter to only passing conversations (all 4 criteria must pass)
    filtered_convs = [uc for uc in unique_convs if uc.judgment.overall_pass]
    logfire.info(
        f"Filtered to {len(filtered_convs)} conversations passing all quality criteria"
    )

    # Sort by number of individual criteria passing (as a tiebreaker, though all should be 4)
    # Then by other factors like naturalness, factual accuracy, etc.
    def quality_score(uc):
        j = uc.judgment
        # All passing conversations have same boolean score, so use criteria count as proxy
        # (though all should be 4/4 if overall_pass is True)
        return sum(
            [j.factually_accurate, j.natural_conversation, j.on_topic, j.adds_value]
        )

    sorted_convs = sorted(filtered_convs, key=quality_score, reverse=True)

    # Select top K
    top_convs = sorted_convs[:top_k]
    logfire.info(f"Selected top {len(top_convs)} conversations")

    # Convert to NanoChat format
    nanochat_convs = [conversation_to_nanochat(uc) for uc in top_convs]

    # Save results
    save_jsonl(nanochat_convs, output_file)
    logfire.info(f"Saved {len(nanochat_convs)} conversations in NanoChat format")

    # Print statistics
    print("\n" + "=" * 80)
    print("TOP-K SELECTION STATISTICS:")
    print("=" * 80)
    print(f"Total unique conversations: {len(unique_convs)}")
    print(f"After minimum score filter: {len(filtered_convs)}")
    print(f"Top K selected: {len(top_convs)}")
    print("=" * 80 + "\n")

    if top_convs:
        passing = sum(1 for uc in top_convs if uc.judgment.overall_pass)
        print("Selected conversation quality:")
        print(
            f"  All passing quality criteria: {passing}/{len(top_convs)} ({passing / len(top_convs) * 100:.1f}%)"
        )
        print("=" * 80 + "\n")

        # Show sample conversation
        sample = top_convs[0]
        print("SAMPLE CONVERSATION:")
        print("=" * 80)
        print(f"Overall pass: {sample.judgment.overall_pass}")
        print(f"Feedback: {sample.judgment.feedback}")
        print("\nQuality criteria:")
        print(f"  Factually accurate: {sample.judgment.factually_accurate}")
        print(f"  Natural conversation: {sample.judgment.natural_conversation}")
        print(f"  On topic: {sample.judgment.on_topic}")
        print(f"  Adds value: {sample.judgment.adds_value}")
        print("\nMessages:")
        for msg in sample.conversation.messages:
            print(f"  {msg.role.upper()}: {msg.content[:100]}...")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
