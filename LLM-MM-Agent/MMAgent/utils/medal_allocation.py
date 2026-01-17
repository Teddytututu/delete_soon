"""
奖牌分配约束工具 (P2 FIX #6)

确保预测的奖牌总数等于给定的池大小

Created: 2026-01-15
Purpose: Fix 114000.05 vs 2280 issue - enforce total pool constraint
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)


def allocate_medals_with_constraint(
    predictions_df: pd.DataFrame,
    prediction_col: str = 'PREDICTED_MEDALS',
    total_pool: float = 2280,
    method: str = 'proportional'
) -> pd.DataFrame:
    """
    分配奖牌到国家/项目，确保总和等于池大小

    Args:
        predictions_df: DataFrame包含预测值
        prediction_col: 预测值列名
        total_pool: 总奖牌池大小 (2028年: 2280)
        method: 分配方法 ('proportional', 'rank-based', 'hybrid')

    Returns:
        添加了 'FINAL_ALLOCATION' 列的DataFrame

    Raises:
        ValueError: 如果输入无效
    """
    df = predictions_df.copy()

    # Validate inputs
    if prediction_col not in df.columns:
        raise ValueError(f"Column '{prediction_col}' not found in DataFrame")

    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    # Step 1: Extract raw predictions
    raw_predictions = df[prediction_col].values

    # Step 2: Ensure non-negative
    raw_predictions = np.maximum(raw_predictions, 0)

    # Step 3: Handle edge cases
    total_raw = raw_predictions.sum()

    if total_raw == 0:
        # Equal allocation
        logger.warning("Sum of predictions is 0, using equal allocation")
        n = len(df)
        final_allocations = np.full(n, total_pool / n)
    else:
        # Step 4: Apply normalization based on method
        if method == 'proportional':
            # Simple proportional allocation
            weights = raw_predictions / total_raw
            final_allocations = weights * total_pool

        elif method == 'rank-based':
            # Allocate based on rank (top gets more)
            # This is useful when predictions are scores, not counts
            ranks = len(df) - df[prediction_col].rank(method='first').values + 1
            # Weight by rank (linear decay)
            rank_weights = ranks / ranks.sum()
            final_allocations = rank_weights * total_pool

        elif method == 'hybrid':
            # Combine proportional and rank-based
            # 70% proportional, 30% rank-based
            weights_prop = raw_predictions / total_raw
            ranks = len(df) - df[prediction_col].rank(method='first').values + 1
            rank_weights = ranks / ranks.sum()

            combined_weights = 0.7 * weights_prop + 0.3 * rank_weights
            combined_weights = combined_weights / combined_weights.sum()  # Re-normalize
            final_allocations = combined_weights * total_pool

        else:
            raise ValueError(f"Unknown method: {method}")

    # Step 5: Verify constraint
    allocation_sum = final_allocations.sum()

    if not np.isclose(allocation_sum, total_pool, rtol=1e-3):
        logger.error(
            f"Constraint verification failed: "
            f"sum={allocation_sum:.2f}, target={total_pool}, "
            f"difference={abs(allocation_sum - total_pool):.2f}"
        )
        # Force exact sum by adjusting the largest value
        idx_largest = np.argmax(final_allocations)
        final_allocations[idx_largest] += (total_pool - allocation_sum)

    # Step 6: Round to integers (medals must be whole numbers)
    # Use sophisticated rounding to preserve sum
    final_integers = np.floor(final_allocations).astype(int)
    remainder = int(total_pool - final_integers.sum())

    # Distribute remainder to those with largest fractional parts
    fractional_parts = final_allocations - final_integers
    idx_top_remainder = np.argsort(-fractional_parts)[:remainder]
    final_integers[idx_top_remainder] += 1

    # Step 7: Update DataFrame
    df['FINAL_ALLOCATION'] = final_integers
    df['RAW_PREDICTION'] = raw_predictions
    df['NORMALIZED_WEIGHT'] = final_allocations / total_pool if total_pool > 0 else 0

    # Logging
    logger.info(f"[P2 FIX #6] Medal allocation completed:")
    logger.info(f"  Method: {method}")
    logger.info(f"  Total predictions: {total_raw:.2f}")
    logger.info(f"  Total pool: {total_pool}")
    logger.info(f"  Final allocation sum: {final_integers.sum()}")
    logger.info(f"  Constraint satisfied: {final_integers.sum() == total_pool}")

    return df


def validate_allocation(
    df: pd.DataFrame,
    allocation_col: str = 'FINAL_ALLOCATION',
    expected_total: float = 2280
) -> Tuple[bool, str]:
    """
    验证分配是否满足约束

    Args:
        df: 包含分配结果的DataFrame
        allocation_col: 分配列名
        expected_total: 期望的总和

    Returns:
        (is_valid, message)
    """
    if allocation_col not in df.columns:
        return False, f"Column '{allocation_col}' not found"

    total = df[allocation_col].sum()

    if not np.isclose(total, expected_total, rtol=1e-3):
        diff = abs(total - expected_total)
        return False, f"Sum {total:.2f} != target {expected_total}, diff={diff:.2f}"

    # Check for negative values
    if (df[allocation_col] < 0).any():
        neg_count = (df[allocation_col] < 0).sum()
        return False, f"Found {neg_count} negative allocations"

    # Check for non-integer values
    if not (df[allocation_col] % 1 == 0).all():
        non_int_count = (df[allocation_col] % 1 != 0).sum()
        return False, f"Found {non_int_count} non-integer allocations"

    return True, f"Valid: sum={total}, all non-negative integers"


# Test code
if __name__ == "__main__":
    print("="*60)
    print("P2 FIX #6: Medal Allocation Constraint Tests")
    print("="*60)

    # Test 1: 正常情况
    print("\nTest 1: Normal case (sum < pool)")
    test_df_1 = pd.DataFrame({
        'NOC': ['USA', 'CHN', 'GBR'],
        'PREDICTED_MEDALS': [100, 80, 60]
    })
    print("Original sum:", test_df_1['PREDICTED_MEDALS'].sum())

    result_1 = allocate_medals_with_constraint(test_df_1, total_pool=2280)
    print("After constraint sum:", result_1['FINAL_ALLOCATION'].sum())
    print("Allocations:", result_1['FINAL_ALLOCATION'].tolist())

    is_valid, msg = validate_allocation(result_1)
    print(f"Validation: {msg}")
    assert is_valid, "Test 1 failed"
    print("  [PASS]")

    # Test 2: 你的日志中的具体案例 - 114000 → 2280
    print("\nTest 2: Your actual problem (sum = 114000)")
    test_df_2 = pd.DataFrame({
        'NOC': ['USA', 'CHN', 'GBR', 'RUS', 'GER'] * 20,
        'PREDICTED_MEDALS': [1140.0] * 100
    })
    print("Original sum:", test_df_2['PREDICTED_MEDALS'].sum())

    result_2 = allocate_medals_with_constraint(test_df_2, total_pool=2280)
    print("After constraint sum:", result_2['FINAL_ALLOCATION'].sum())

    is_valid, msg = validate_allocation(result_2)
    print(f"Validation: {msg}")
    assert is_valid, "Test 2 failed"
    print("  [PASS]")

    print("\n" + "="*60)
    print("[SUCCESS] All P2 FIX #6 tests passed! ✅")
    print("114000.05 → 2280.00 constraint works correctly!")
    print("="*60)
