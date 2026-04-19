import yaml
import torch
from unsloth import FastLanguageModel

class IntentClassification:
    def __init__(self, model_path):
        """
        Loads the configuration file, tokenizer, and model checkpoint.
        """
        print(f"Loading configuration from {model_path}...")
        with open(model_path, "r") as file:
            self.config = yaml.safe_load(file)

        checkpoint_dir = self.config["checkpoint_path"]
        max_seq_length = self.config.get("max_seq_length", 128)

        print(f"Loading model and tokenizer from {checkpoint_dir}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = checkpoint_dir,
            max_seq_length = max_seq_length,
            dtype = None,
            load_in_4bit = True,
        )

        # Enable Unsloth's native inference optimisation
        FastLanguageModel.for_inference(self.model)

        # Predefine the label mapping to translate the model's output ID back to a string
        self.intent_labels = [
            "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
            "automatic_top_up", "balance_not_updated_after_bank_transfer",
            "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
            "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
            "card_delivery_estimate", "card_linking", "card_not_working",
            "card_payment_fee_charged", "card_payment_not_recognised",
            "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
            "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
            "contactless_not_working", "country_support", "declined_card_payment",
            "declined_cash_withdrawal", "declined_transfer",
            "direct_debit_payment_not_recognised", "disposable_card_limits",
            "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
            "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
            "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
            "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
            "order_physical_card", "passcode_forgotten", "pending_card_payment",
            "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
            "receiving_money", "Refund_not_showing_up", "request_refund",
            "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
            "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
            "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
            "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
            "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
            "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
            "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
            "why_verify_identity", "wrong_amount_of_cash_received",
            "wrong_exchange_rate_for_cash_withdrawal"
        ]

    def __call__(self, message):
        """
        Receives an input message and returns the predicted intent label.
        """
        # Format the input exactly as it was formatted during training
        prompt = f"Categorise the intent of the following banking query.\n\nQuery: {message}\n\nIntent ID:"

        # Tokenise the input string
        inputs = self.tokenizer(
            [prompt], return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate the prediction
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=5, # Restrict to a few tokens to yield only the integer ID
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode the generated output back into standard text
        decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Parse the predicted integer ID and map it back to the respective string label
        try:
            # Isolate the generated number immediately following "Intent ID:"
            predicted_id_str = decoded_output.split("Intent ID:")[-1].strip()
            # Extract only the digits in case the model generated any spurious characters
            predicted_id = int(''.join(filter(str.isdigit, predicted_id_str)))
            predicted_label = self.intent_labels[predicted_id]
        except (ValueError, IndexError):
            predicted_label = "unknown_intent"

        return predicted_label


if __name__ == "__main__":
    # Define the path to your configuration file
    config_path = "configs/inference.yaml"
    
    # Initialise the classifier (this implicitly calls the __init__ method)
    classifier = IntentClassification(model_path=config_path)
    
    # Define a test message
    test_message = "I am still waiting on my card, when will it arrive?"
    
    # Predict the label (this implicitly calls the __call__ method)
    print(f"\nInput Message: '{test_message}'")
    prediction = classifier(test_message)
    print(f"Predicted Label: {prediction}")