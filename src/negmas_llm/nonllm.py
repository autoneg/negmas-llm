import random

from negmas.common import Outcome
from negmas.gb import (
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    GBNegotiator,
    LinearTBNegotiator,
)
from negmas.gb.common import ExtendedResponseType, ResponseType
from negmas.gb.negotiators.hybrid import HybridNegotiator
from negmas.outcomes import ExtendedOutcome
from negmas.sao import SAONegotiator, SAOState
from negmas.sao.negotiators.meta import SAOMetaNegotiator

__all__ = [
    "TemplateBasedAdapterNegotiator",
    "BoulwareWithTextNegotiator",
    "ConcederWithTextNegotiator",
    "LinearWithTextNegotiator",
    "HybridWithTextNegotiator",
    "ACCEPTANCE_MESSAGES",
    "CHANGE_PHRASES",
    "COMPARISON_WORDS",
    "REJECTION_STARTERS",
    "REJECTION_ENDERS",
    "OPENING_OFFER_STARTERS",
    "OPENING_OFFER_ENDERS",
]

# Predefined message templates for acceptance
ACCEPTANCE_MESSAGES = [
    "Excellent! I'm pleased to accept your offer. This works well for both of us.",
    "Thank you for your flexibility. I'm happy to accept this proposal.",
    "I appreciate your efforts in reaching this agreement. I accept your offer.",
    "This is a fair deal. I'm delighted to accept and conclude our negotiation.",
    "Perfect! Your offer aligns with my interests. I gladly accept.",
    "I'm satisfied with these terms. Let's seal the deal.",
    "Great! This is exactly what I was hoping for. I accept.",
    "Wonderful! I believe this is a win-win solution. I accept your offer.",
    "I'm very pleased with this outcome. Thank you for a productive negotiation.",
    "This works perfectly for me. I'm happy to move forward with this agreement.",
    "Agreed! I think we've reached a mutually beneficial arrangement.",
    "Outstanding! I accept these terms without hesitation.",
]

# Predefined templates for describing differences
CHANGE_PHRASES = [
    "{issue} of {their_value} is {comparison}, I'm proposing {my_value} instead",
    "I believe {issue} should be {my_value} rather than {their_value}",
    "{their_value} for {issue} is {comparison}, so I'm suggesting {my_value}",
    "I'm proposing {my_value} for {issue} instead of your {their_value}",
    "your {their_value} for {issue} is {comparison}; I'd prefer {my_value}",
    "regarding {issue}, {their_value} is {comparison}; {my_value} would work better",
    "I think {my_value} for {issue} is more reasonable than {their_value}",
    "the {issue} value of {their_value} is {comparison}; {my_value} is fairer",
    "{issue} at {their_value} is {comparison}, I'm countering with {my_value}",
    "I need {my_value} for {issue} because {their_value} is {comparison}",
    "for {issue}, I'm offering {my_value} as {their_value} is {comparison}",
    "let's adjust {issue} from {their_value} to {my_value} since yours is {comparison}",
]

COMPARISON_WORDS = {
    "higher": [
        "too high",
        "more than I can accept",
        "above my target",
        "beyond what I consider fair",
        "higher than reasonable",
        "exceeding my limits",
        "more than the value warrants",
        "steeper than I anticipated",
    ],
    "lower": [
        "too low",
        "less than what's fair",
        "below what I need",
        "under my minimum threshold",
        "insufficient for my needs",
        "lower than I can accommodate",
        "not meeting my requirements",
        "below market value",
    ],
    "different": [
        "not quite right",
        "not aligned with my needs",
        "not suitable",
        "not matching my preferences",
        "incompatible with my objectives",
        "not what I had in mind",
        "diverging from my expectations",
        "not optimal for this agreement",
    ],
}

# Sentence starters for rejection/counteroffer messages
REJECTION_STARTERS = [
    "I appreciate your offer, but",
    "Thank you for your proposal, however",
    "While I understand your position,",
    "I've considered your offer carefully, but",
    "I see where you're coming from, however",
    "Thanks for the suggestion, though",
    "I respect your proposal, but",
    "That's an interesting offer, however",
    "I value your input, but",
    "After careful consideration,",
    "Looking at your proposal,",
    "I've reviewed your terms, and",
]

# Sentence enders for rejection/counteroffer messages (optional)
REJECTION_ENDERS = [
    "I hope this works better for both of us.",
    "Let me know if this seems more reasonable.",
    "I believe this is a fairer arrangement.",
    "I think this better reflects our mutual interests.",
    "This should address our concerns more effectively.",
    "I trust you'll find this more acceptable.",
    "Perhaps we can find common ground here.",
    "I'm confident this is a step in the right direction.",
    "This might serve both our needs better.",
    "I'm hopeful we can agree on this.",
    "Let's see if this works for you.",
    "I'm open to further discussion on this.",
]

# Opening offer messages (for the first proposal when no opponent offer exists)
OPENING_OFFER_STARTERS = [
    "Thank you for the opportunity to negotiate. Let me start with",
    "I'm pleased to begin our negotiation. Here's my initial proposal:",
    "Let's get started! I'd like to propose",
    "I appreciate the chance to discuss terms. My opening offer is",
    "Hello! I'd like to kick things off with",
    "Great to meet you! Here's what I have in mind:",
    "Let me begin by proposing",
    "To start our discussion, I'd like to suggest",
    "I'm excited to negotiate with you. My first offer is",
    "Let's find a mutually beneficial agreement. I propose",
    "Thank you for negotiating with me. I'd like to start with",
    "Here's my opening position:",
]

OPENING_OFFER_ENDERS = [
    "I'm open to hearing your thoughts.",
    "Let me know what you think.",
    "I look forward to your response.",
    "I hope this is a good starting point.",
    "I'm eager to hear your perspective.",
    "What do you think?",
    "I believe this is a fair starting point.",
    "Let's work together to find common ground.",
    "I'm confident we can reach an agreement.",
    "Looking forward to a productive negotiation.",
]


class TemplateBasedAdapterNegotiator(SAOMetaNegotiator):
    """A negotiator that adapts a base negotiator by adding predefined text messages.

    This negotiator wraps any base negotiator and enhances its responses with
    appropriate text messages based on the action taken (accept, reject, counteroffer).
    Unlike LLMMetaNegotiator, this uses predefined templates instead of LLM calls.
    """

    def __init__(
        self, base_negotiator: GBNegotiator | SAONegotiator | None = None, **kwargs
    ) -> None:
        """Initialize the TemplateBasedAdapterNegotiator.

        Args:
            base_negotiator: The base negotiator to adapt.
                If None, creates a BoulwareTBNegotiator.
            **kwargs: Additional arguments passed to SAOMetaNegotiator.
        """
        # Use default Boulware strategy if no base negotiator provided
        if base_negotiator is None:
            base_negotiator = BoulwareTBNegotiator()

        # Initialize with the base negotiator as our single child
        super().__init__(
            negotiators=[base_negotiator],  # type: ignore
            negotiator_names=["base"],
            share_ufun=True,
            share_nmi=True,
            **kwargs,
        )

    @property
    def base_negotiator(self) -> GBNegotiator:
        """The underlying negotiator that handles core negotiation logic."""
        return self._negotiators[0]  # type: ignore

    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Get proposal from base negotiator and add template-generated text.

        Args:
            state: The current SAO state.
            dest: The destination partner ID (if applicable).

        Returns:
            An ExtendedOutcome with the base proposal and template-generated text.
        """
        # Get proposal from base negotiator
        base_proposal = self.base_negotiator.propose(state, dest=dest)

        if base_proposal is None:
            return None

        # Extract the outcome if it's already an ExtendedOutcome
        if isinstance(base_proposal, ExtendedOutcome):
            outcome = base_proposal.outcome
            base_data = base_proposal.data or {}
        else:
            outcome = base_proposal
            base_data = {}

        if outcome is None:
            return None

        # Generate text to accompany the offer
        generated_text = self._generate_text(state, "propose", outcome, None)

        # Combine base data with generated text
        data = {**base_data, "text": generated_text}

        return ExtendedOutcome(outcome=outcome, data=data)

    def respond(
        self, state: SAOState, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:  # type: ignore
        """Get response from base negotiator and add template-generated text.

        Args:
            state: The current SAO state.
            source: The source partner ID.

        Returns:
            An ExtendedResponseType with the base response and template-generated text.
        """
        # Get response from base negotiator
        base_response = self.base_negotiator.respond(state, source=source)

        # Extract the response type if it's already an ExtendedResponseType
        if isinstance(base_response, ExtendedResponseType):
            response_type = base_response.response
            base_data = base_response.data or {}
        else:
            response_type = base_response
            base_data = {}

        # Determine the action for text generation
        if response_type == ResponseType.ACCEPT_OFFER:
            action = "accept"
        elif response_type == ResponseType.END_NEGOTIATION:
            action = "end"
        else:
            # For rejections, return without text (added with counter-proposal)
            return base_response

        # Generate text to accompany the response
        generated_text = self._generate_text(state, action, state.current_offer, None)

        # Combine base data with generated text
        data = {**base_data, "text": generated_text}

        return ExtendedResponseType(response=response_type, data=data)

    def _generate_text(
        self,
        state: SAOState,
        action: str,
        outcome: Outcome | None = None,
        received_text: str | None = None,
    ) -> str:
        """Generate text to accompany an offer or response using predefined templates.

        Args:
            state: The current negotiation state.
            action: The action being taken ("propose", "accept", "reject", "end").
            outcome: The outcome being proposed (if any).
            received_text: Text received from the other party (if any).

        Returns:
            The generated text message.
        """
        if action == "accept":
            return self._generate_acceptance_message(state, received_text)
        elif action == "reject" or action == "propose":
            return self._generate_rejection_and_counteroffer_message(
                state, outcome, received_text
            )
        elif action == "end":
            return "I believe we've reached an impasse. Thank you for your time."
        else:
            return ""

    def _generate_acceptance_message(
        self,
        state: SAOState,
        received_text: str | None = None,
    ) -> str:
        """Generate a nice acceptance message with specific details.

        Args:
            state: The current negotiation state.
            received_text: Text received from the opponent.

        Returns:
            An acceptance message.
        """
        base_message = random.choice(ACCEPTANCE_MESSAGES)

        # Add specific details about what we're accepting if available
        if state.current_offer is not None and self.nmi is not None:
            outcome_space = self.nmi.outcome_space
            if outcome_space:
                issues = getattr(outcome_space, "issues", None)
                if issues:
                    # Mention a few key issues in the acceptance
                    details = []
                    for i, issue in enumerate(issues[:2]):  # Mention up to 2 issues
                        if i < len(state.current_offer):
                            value = state.current_offer[i]
                            details.append(f"{issue.name}={value}")
                    if details:
                        base_message += f" ({', '.join(details)})"

        return base_message

    def _generate_rejection_and_counteroffer_message(
        self,
        state: SAOState,
        outcome: Outcome | None,
        received_text: str | None = None,
    ) -> str:
        """Generate a contextual rejection message with specific comparisons.

        Args:
            state: The current negotiation state.
            outcome: The counteroffer being proposed.
            received_text: Text received from the opponent.

        Returns:
            A message explaining the rejection and counteroffer with issue values.
        """
        their_offer = state.current_offer
        my_offer = outcome

        # Check if this is an opening offer (no opponent offer yet)
        if their_offer is None:
            return self._generate_opening_offer_message(state, my_offer)

        if my_offer is None or self.nmi is None:
            # Fallback to generic message
            return (
                "I cannot accept your offer. "
                "Let me propose an alternative that better suits both our needs."
            )

        outcome_space = self.nmi.outcome_space
        if not outcome_space:
            return "I'm proposing a counteroffer that I believe is more balanced."

        issues = getattr(outcome_space, "issues", None)
        if not issues:
            return "I'm proposing a counteroffer that I believe is more balanced."

        differences = []

        # Find issues where our offer differs from theirs
        for i, issue in enumerate(issues):
            if i >= len(their_offer) or i >= len(my_offer):
                continue

            their_value = their_offer[i]
            my_value = my_offer[i]

            if their_value != my_value:
                # Determine comparison type
                try:
                    # Try to compare numerically
                    their_num = float(their_value)
                    my_num = float(my_value)
                    if my_num > their_num:
                        comparison = random.choice(COMPARISON_WORDS["lower"])
                    elif my_num < their_num:
                        comparison = random.choice(COMPARISON_WORDS["higher"])
                    else:
                        comparison = random.choice(COMPARISON_WORDS["different"])
                except (ValueError, TypeError):
                    # Not numeric, just say different
                    comparison = random.choice(COMPARISON_WORDS["different"])

                # Create a description of this difference
                phrase_template = random.choice(CHANGE_PHRASES)
                difference_desc = phrase_template.format(
                    issue=issue.name,
                    their_value=their_value,
                    my_value=my_value,
                    comparison=comparison,
                )
                differences.append(difference_desc)

        # Construct the message
        if not differences:
            return (
                "I'd like to propose a slightly different arrangement "
                "that works better for me."
            )

        # Use 1-2 most important differences
        num_to_mention = min(2, len(differences))
        selected_diffs = differences[:num_to_mention]

        # Select a random starter and optionally an ender
        starter = random.choice(REJECTION_STARTERS)

        # Build the core message
        if len(selected_diffs) == 1:
            core_message = f"{selected_diffs[0]}"
        else:
            core_message = f"{selected_diffs[0]}, and {selected_diffs[1]}"

        # Randomly decide whether to add an ender (50% chance)
        if random.random() < 0.5:
            ender = random.choice(REJECTION_ENDERS)
            return f"{starter} {core_message}. {ender}"
        else:
            return f"{starter} {core_message}."

    def _generate_opening_offer_message(
        self,
        state: SAOState,
        my_offer: Outcome | None,
    ) -> str:
        """Generate a message for the opening offer (when no opponent offer exists yet).

        Args:
            state: The current negotiation state.
            my_offer: The offer being proposed.

        Returns:
            A friendly opening message describing the initial proposal.
        """
        starter = random.choice(OPENING_OFFER_STARTERS)

        # Try to describe the offer details
        if my_offer is not None and self.nmi is not None:
            outcome_space = self.nmi.outcome_space
            if outcome_space:
                issues = getattr(outcome_space, "issues", None)
                if issues:
                    # Describe 1-2 key issues from the offer
                    details = []
                    for i, issue in enumerate(issues[:2]):
                        if i < len(my_offer):
                            value = my_offer[i]
                            details.append(f"{issue.name}={value}")
                    if details:
                        offer_desc = ", ".join(details)
                        # Randomly add an ender (50% chance)
                        if random.random() < 0.5:
                            ender = random.choice(OPENING_OFFER_ENDERS)
                            return f"{starter} {offer_desc}. {ender}"
                        else:
                            return f"{starter} {offer_desc}."

        # Fallback without specific details
        ender = random.choice(OPENING_OFFER_ENDERS)
        return f"{starter} {ender}"


class BoulwareWithTextNegotiator(TemplateBasedAdapterNegotiator):
    """Boulware (tough) negotiator with template-based text messages.

    This negotiator starts with high demands and concedes slowly,
    making most concessions near the deadline. Uses predefined templates
    for generating human-readable text messages.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize with a BoulwareTBNegotiator as the base strategy.

        Args:
            **kwargs: Additional arguments passed to TemplateBasedAdapterNegotiator.
        """
        super().__init__(base_negotiator=BoulwareTBNegotiator(), **kwargs)


class ConcederWithTextNegotiator(TemplateBasedAdapterNegotiator):
    """Conceder (soft) negotiator with template-based text messages.

    This negotiator concedes quickly early in the negotiation,
    making most concessions at the beginning. Uses predefined templates
    for generating human-readable text messages.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize with a ConcederTBNegotiator as the base strategy.

        Args:
            **kwargs: Additional arguments passed to TemplateBasedAdapterNegotiator.
        """
        super().__init__(base_negotiator=ConcederTBNegotiator(), **kwargs)


class LinearWithTextNegotiator(TemplateBasedAdapterNegotiator):
    """Linear negotiator with template-based text messages.

    This negotiator concedes at a constant rate throughout the negotiation.
    Uses predefined templates for generating human-readable text messages.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize with a LinearTBNegotiator as the base strategy.

        Args:
            **kwargs: Additional arguments passed to TemplateBasedAdapterNegotiator.
        """
        super().__init__(base_negotiator=LinearTBNegotiator(), **kwargs)


class HybridWithTextNegotiator(TemplateBasedAdapterNegotiator):
    """Hybrid negotiator with template-based text messages.

    A hybrid negotiator that combines multiple strategies using HybridOfferingPolicy
    and ACNext acceptance policy, with template-based text accompanying each offer.
    Unlike LLMHybridNegotiator, this uses predefined templates instead of LLM calls.

    Args:
        alpha: Parameter for the ACNext acceptance policy (default: 1.0).
            Controls the acceptance threshold relative to the next expected offer.
        beta: Parameter for the ACNext acceptance policy (default: 0.0).
            Adds a constant offset to the acceptance threshold.
        **kwargs: Additional arguments passed to TemplateBasedAdapterNegotiator.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        **kwargs,
    ) -> None:
        """Initialize with a HybridNegotiator as the base strategy.

        Args:
            alpha: Parameter for the ACNext acceptance policy (default: 1.0).
            beta: Parameter for the ACNext acceptance policy (default: 0.0).
            **kwargs: Additional arguments passed to TemplateBasedAdapterNegotiator.
        """
        base_negotiator = HybridNegotiator(alpha=alpha, beta=beta)
        super().__init__(base_negotiator=base_negotiator, **kwargs)
