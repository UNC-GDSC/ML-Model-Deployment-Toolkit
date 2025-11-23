"""
Cost Estimation Example

This example demonstrates how to estimate and compare deployment costs
across different cloud platforms.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cost.cost_estimator import CostEstimator


def basic_cost_estimation():
    """Basic cost estimation for a single platform."""
    print("=" * 70)
    print("Basic Cost Estimation - AWS Lambda")
    print("=" * 70)
    print()

    estimator = CostEstimator()

    # Estimate Lambda costs
    estimate = estimator.estimate_lambda(
        requests_per_month=1_000_000,  # 1 million requests/month
        avg_duration_ms=200,  # 200ms average duration
        memory_mb=1024  # 1GB memory
    )

    print(f"Platform: {estimate.platform}")
    print(f"Monthly Cost: ${estimate.monthly_cost_usd:.2f}")
    print(f"Cost per 1K requests: ${estimate.cost_per_1k_requests:.4f}")
    print(f"Cost per 1M requests: ${estimate.cost_per_1m_requests:.2f}")
    print()

    print("Cost Breakdown:")
    for component, cost in estimate.breakdown.items():
        print(f"  {component}: ${cost:.2f}")
    print()


def platform_comparison():
    """Compare costs across multiple platforms."""
    print("=" * 70)
    print("Platform Cost Comparison")
    print("=" * 70)
    print()

    estimator = CostEstimator()

    # Compare platforms
    scenarios = [
        ("Low Traffic", 100_000, 150),
        ("Medium Traffic", 1_000_000, 200),
        ("High Traffic", 10_000_000, 250),
    ]

    for scenario_name, requests, duration in scenarios:
        print(f"\nScenario: {scenario_name}")
        print(f"  Requests/month: {requests:,}")
        print(f"  Avg duration: {duration}ms")
        print("-" * 70)

        estimates = estimator.compare_platforms(
            requests_per_month=requests,
            avg_duration_ms=duration,
            memory_gb=1.0
        )

        print(f"{'Platform':<20} {'Monthly Cost':<15} {'Cost/1M Requests':<20}")
        print("-" * 70)

        for est in estimates:
            print(f"{est.platform:<20} ${est.monthly_cost_usd:<14.2f} ${est.cost_per_1m_requests:<19.2f}")

        print()


def cost_optimization_recommendations():
    """Get platform recommendations based on workload."""
    print("=" * 70)
    print("Cost Optimization Recommendations")
    print("=" * 70)
    print()

    estimator = CostEstimator()

    workloads = [
        {
            "name": "Startup MVP",
            "requests": 50_000,
            "duration": 150,
            "memory": 0.5,
            "requirements": {}
        },
        {
            "name": "Production API",
            "requests": 5_000_000,
            "duration": 200,
            "memory": 1.0,
            "requirements": {}
        },
        {
            "name": "ML Inference (GPU)",
            "requests": 1_000_000,
            "duration": 500,
            "memory": 4.0,
            "requirements": {"gpu_required": True}
        }
    ]

    for workload in workloads:
        print(f"\nWorkload: {workload['name']}")
        print("-" * 70)

        recommendation = estimator.recommend_platform(
            requests_per_month=workload['requests'],
            avg_duration_ms=workload['duration'],
            memory_gb=workload['memory'],
            requirements=workload['requirements']
        )

        print(f"Recommended Platform: {recommendation['recommended_platform']}")
        print(f"Estimated Monthly Cost: ${recommendation['estimated_monthly_cost']:.2f}")
        print(f"Cost per 1M Requests: ${recommendation['cost_per_1m_requests']:.2f}")
        print(f"Reasoning: {recommendation['reasoning']}")
        print()


def detailed_sagemaker_analysis():
    """Detailed cost analysis for SageMaker deployment."""
    print("=" * 70)
    print("AWS SageMaker Cost Analysis")
    print("=" * 70)
    print()

    estimator = CostEstimator()

    instance_types = [
        "ml.t2.medium",
        "ml.m5.large",
        "ml.c5.xlarge",
        "ml.g4dn.xlarge"  # GPU instance
    ]

    requests_per_month = 1_000_000

    print(f"Monthly cost for {requests_per_month:,} requests/month\n")
    print(f"{'Instance Type':<20} {'Hourly Rate':<15} {'Monthly Cost':<15} {'Cost/1M Req':<15}")
    print("-" * 70)

    for instance_type in instance_types:
        estimate = estimator.estimate_sagemaker(
            instance_type=instance_type,
            num_instances=1,
            requests_per_month=requests_per_month
        )

        hourly_rate = estimator.PRICING["aws_sagemaker"][instance_type]

        print(f"{instance_type:<20} ${hourly_rate:<14.3f} ${estimate.monthly_cost_usd:<14.2f} ${estimate.cost_per_1m_requests:<14.2f}")

    print()


def cost_projections():
    """Project costs for different growth scenarios."""
    print("=" * 70)
    print("Cost Projections - Growth Scenarios")
    print("=" * 70)
    print()

    estimator = CostEstimator()

    months = [1, 3, 6, 12]
    growth_rate = 1.5  # 50% growth per month

    print("AWS Lambda Cost Projections (50% monthly growth)")
    print()
    print(f"{'Month':<10} {'Requests':<15} {'Monthly Cost':<15} {'Cumulative Cost':<15}")
    print("-" * 70)

    cumulative_cost = 0
    initial_requests = 100_000

    for month in months:
        requests = int(initial_requests * (growth_rate ** (month - 1)))

        estimate = estimator.estimate_lambda(
            requests_per_month=requests,
            avg_duration_ms=200,
            memory_mb=1024
        )

        cumulative_cost += estimate.monthly_cost_usd

        print(f"{month:<10} {requests:<15,} ${estimate.monthly_cost_usd:<14.2f} ${cumulative_cost:<14.2f}")

    print()


def main():
    """Run all examples."""
    basic_cost_estimation()
    platform_comparison()
    cost_optimization_recommendations()
    detailed_sagemaker_analysis()
    cost_projections()

    print("=" * 70)
    print("Cost Estimation Complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("1. Serverless (Lambda, Cloud Run) is cost-effective for variable workloads")
    print("2. SageMaker provides managed ML infrastructure at premium pricing")
    print("3. Kubernetes offers best value at high, consistent traffic")
    print("4. Always consider free tiers when starting out")
    print("5. Monitor actual usage and adjust instance types accordingly")
    print()


if __name__ == '__main__':
    main()
