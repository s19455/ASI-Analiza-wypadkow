#!/usr/bin/env python
"""
Verification script for crash_encoders artifact.

This script demonstrates that:
1. Encoders can be loaded from the pickle file
2. Encoders contain mappings for all categorical columns
3. Transform can be applied to new data using loaded encoders
"""

import pandas as pd
from crash_kedro.utils.encoders import load_encoders, transform_with_encoders


def main():
    """Load and verify crash_encoders artifact."""
    # Load encoders from disk
    encoders_path = "data/06_models/encoders.pkl"
    print(f"Loading encoders from {encoders_path}...")
    encoders = load_encoders(encoders_path)
    
    print(f"\n✅ Successfully loaded {len(encoders)} encoders\n")
    
    # Show details for first 5 encoders
    print("Sample encoder details:")
    print("-" * 60)
    for i, (col_name, encoder_mapping) in enumerate(list(encoders.items())[:5]):
        num_categories = len(encoder_mapping)
        sample_values = list(encoder_mapping.items())[:3]
        print(f"\n{i+1}. Column: '{col_name}'")
        print(f"   Categories: {num_categories}")
        print(f"   Sample mappings: {sample_values}")
    
    # Create sample inference data
    print("\n" + "=" * 60)
    print("Testing with sample inference data:")
    print("=" * 60)
    
    sample_data = pd.DataFrame({
        "Route Type": ["STATE ROUTE", "COUNTY ROAD", "COUNTY ROAD"],
        "Weather": ["CLEAR", "RAIN", "UNKNOWN_WEATHER"],  # Unknown category
        "Light": ["DAY", "DARK", "DARK"],
        "Traffic Control": ["TRAFFIC SIGNAL", "STOP SIGN", "YIELD"],
        "Vehicle Make": ["FORD", "HONDA", "UNKNOWN_MAKE"],  # Unknown category
        "Severity_Group": ["NO_INJURY", "INJURY", "FATALITY"],  # Not encoded
    })
    
    # Add remaining columns that were in training
    # (in real scenario, you'd have all columns)
    for col in encoders:
        if col not in sample_data.columns:
            sample_data[col] = "PLACEHOLDER"
    
    print("\nOriginal sample data (first 3 rows):")
    print(sample_data[["Route Type", "Weather", "Light", "Vehicle Make", "Severity_Group"]].head(3))
    
    # Transform using loaded encoders
    print("\nTransforming with loaded encoders...")
    transformed_data = transform_with_encoders(sample_data, encoders)
    
    print("\nTransformed sample data (first 3 rows):")
    print(transformed_data[["Route Type", "Weather", "Light", "Vehicle Make", "Severity_Group"]].head(3))
    
    print("\n✅ Transformation successful!")
    print("\nKey observations:")
    print(f"  - 'Weather' column with 'RAIN' → {transformed_data.loc[1, 'Weather']:.0f}")
    print(f"  - 'Weather' column with 'UNKNOWN_WEATHER' → {transformed_data.loc[2, 'Weather']:.0f} (unknown category code)")
    print(f"  - 'Vehicle Make' with 'UNKNOWN_MAKE' → {transformed_data.loc[2, 'Vehicle Make']:.0f} (unknown category code)")
    print(f"  - 'Severity_Group' preserved as-is: {transformed_data.loc[0, 'Severity_Group']}")


if __name__ == "__main__":
    main()

