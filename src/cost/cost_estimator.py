"""
Cost estimation and optimization for ML model deployments.

Estimates deployment costs across different platforms and provides
optimization recommendations.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    """Cost estimate for a deployment."""
    platform: str
    monthly_cost_usd: float
    cost_per_1k_requests: float
    cost_per_1m_requests: float
    breakdown: Dict[str, float]
    assumptions: Dict[str, Any]


class CostEstimator:
    """
    Estimate deployment costs across platforms.

    Calculates costs for AWS Lambda, SageMaker, GCP Cloud Run,
    Kubernetes, Azure Functions, and Vercel.
    """

    # Platform pricing (as of 2024, approximate)
    PRICING = {
        "aws_lambda": {
            "request_cost": 0.20 / 1_000_000,  # $0.20 per 1M requests
            "compute_cost_per_gb_second": 0.0000166667,  # $0.0000166667 per GB-second
            "free_tier_requests": 1_000_000,  # 1M requests/month
            "free_tier_compute": 400_000  # 400,000 GB-seconds/month
        },
        "aws_sagemaker": {
            "ml.t2.medium": 0.065,  # $/hour
            "ml.t2.large": 0.130,
            "ml.m5.large": 0.115,
            "ml.m5.xlarge": 0.230,
            "ml.c5.xlarge": 0.204,
            "ml.c5.2xlarge": 0.408,
            "ml.g4dn.xlarge": 0.736,  # GPU instance
            "data_transfer": 0.09  # $/GB
        },
        "gcp_cloud_run": {
            "request_cost": 0.40 / 1_000_000,  # $0.40 per 1M requests
            "cpu_cost_per_second": 0.00002400 / 1,  # per vCPU-second
            "memory_cost_per_gb_second": 0.00000250,
            "free_tier_requests": 2_000_000,  # 2M requests/month
            "free_tier_cpu_hours": 180_000,  # CPU-seconds/month
            "free_tier_memory_gb_hours": 360_000  # GB-seconds/month
        },
        "kubernetes": {
            "node_costs": {
                "n1-standard-2": 0.095,  # $/hour (GCP)
                "n1-standard-4": 0.190,
                "t3.medium": 0.0416,  # $/hour (AWS)
                "t3.large": 0.0832
            },
            "load_balancer": 0.025,  # $/hour
            "persistent_disk_gb": 0.040  # $/GB-month
        },
        "azure_functions": {
            "execution_cost_per_gb_second": 0.000016,
            "request_cost": 0.20 / 1_000_000,
            "free_tier_executions": 1_000_000,
            "free_tier_gb_seconds": 400_000
        },
        "vercel": {
            "hobby_plan": 0,  # Free tier
            "pro_plan": 20,  # $/month
            "enterprise_plan": 500,  # $/month (custom)
            "gb_hour_cost": 0.18  # $/GB-hour beyond included
        }
    }

    def __init__(self):
        """Initialize cost estimator."""
        self.estimates = []

    def estimate_lambda(
        self,
        requests_per_month: int,
        avg_duration_ms: int,
        memory_mb: int = 1024
    ) -> CostEstimate:
        """
        Estimate AWS Lambda costs.

        Args:
            requests_per_month: Number of requests per month
            avg_duration_ms: Average request duration in milliseconds
            memory_mb: Memory allocation in MB

        Returns:
            Cost estimate
        """
        pricing = self.PRICING["aws_lambda"]

        # Calculate compute time
        gb_seconds = (memory_mb / 1024) * (avg_duration_ms / 1000) * requests_per_month

        # Apply free tier
        billable_requests = max(0, requests_per_month - pricing["free_tier_requests"])
        billable_gb_seconds = max(0, gb_seconds - pricing["free_tier_compute"])

        # Calculate costs
        request_cost = billable_requests * pricing["request_cost"]
        compute_cost = billable_gb_seconds * pricing["compute_cost_per_gb_second"]

        total_cost = request_cost + compute_cost

        return CostEstimate(
            platform="AWS Lambda",
            monthly_cost_usd=total_cost,
            cost_per_1k_requests=(total_cost / requests_per_month * 1000) if requests_per_month > 0 else 0,
            cost_per_1m_requests=(total_cost / requests_per_month * 1_000_000) if requests_per_month > 0 else 0,
            breakdown={
                "request_cost": request_cost,
                "compute_cost": compute_cost,
                "free_tier_savings": (
                    (requests_per_month - billable_requests) * pricing["request_cost"] +
                    (gb_seconds - billable_gb_seconds) * pricing["compute_cost_per_gb_second"]
                )
            },
            assumptions={
                "requests_per_month": requests_per_month,
                "avg_duration_ms": avg_duration_ms,
                "memory_mb": memory_mb,
                "includes_free_tier": True
            }
        )

    def estimate_sagemaker(
        self,
        instance_type: str = "ml.t2.medium",
        num_instances: int = 1,
        requests_per_month: int = 1_000_000
    ) -> CostEstimate:
        """
        Estimate AWS SageMaker costs.

        Args:
            instance_type: SageMaker instance type
            num_instances: Number of instances
            requests_per_month: Requests per month (for context)

        Returns:
            Cost estimate
        """
        pricing = self.PRICING["aws_sagemaker"]

        if instance_type not in pricing:
            instance_type = "ml.t2.medium"
            logger.warning(f"Unknown instance type, using {instance_type}")

        # Instance costs (24/7)
        instance_cost_per_hour = pricing[instance_type]
        hours_per_month = 730  # Average hours per month

        instance_cost = instance_cost_per_hour * hours_per_month * num_instances

        # Data transfer (estimate 10KB per request in + out)
        data_transfer_gb = (requests_per_month * 20 / 1024 / 1024)  # 20KB total per request
        transfer_cost = data_transfer_gb * pricing["data_transfer"]

        total_cost = instance_cost + transfer_cost

        return CostEstimate(
            platform="AWS SageMaker",
            monthly_cost_usd=total_cost,
            cost_per_1k_requests=(total_cost / requests_per_month * 1000) if requests_per_month > 0 else 0,
            cost_per_1m_requests=(total_cost / requests_per_month * 1_000_000) if requests_per_month > 0 else 0,
            breakdown={
                "instance_cost": instance_cost,
                "data_transfer_cost": transfer_cost
            },
            assumptions={
                "instance_type": instance_type,
                "num_instances": num_instances,
                "uptime": "24/7",
                "requests_per_month": requests_per_month
            }
        )

    def estimate_cloud_run(
        self,
        requests_per_month: int,
        avg_duration_ms: int,
        cpu: float = 1.0,
        memory_gb: float = 1.0
    ) -> CostEstimate:
        """
        Estimate GCP Cloud Run costs.

        Args:
            requests_per_month: Requests per month
            avg_duration_ms: Average duration in milliseconds
            cpu: CPU allocation (vCPUs)
            memory_gb: Memory allocation in GB

        Returns:
            Cost estimate
        """
        pricing = self.PRICING["gcp_cloud_run"]

        # CPU and memory time
        cpu_seconds = cpu * (avg_duration_ms / 1000) * requests_per_month
        memory_gb_seconds = memory_gb * (avg_duration_ms / 1000) * requests_per_month

        # Apply free tier
        billable_requests = max(0, requests_per_month - pricing["free_tier_requests"])
        billable_cpu_seconds = max(0, cpu_seconds - pricing["free_tier_cpu_hours"])
        billable_memory_gb_seconds = max(0, memory_gb_seconds - pricing["free_tier_memory_gb_hours"])

        # Calculate costs
        request_cost = billable_requests * pricing["request_cost"]
        cpu_cost = billable_cpu_seconds * pricing["cpu_cost_per_second"]
        memory_cost = billable_memory_gb_seconds * pricing["memory_cost_per_gb_second"]

        total_cost = request_cost + cpu_cost + memory_cost

        return CostEstimate(
            platform="GCP Cloud Run",
            monthly_cost_usd=total_cost,
            cost_per_1k_requests=(total_cost / requests_per_month * 1000) if requests_per_month > 0 else 0,
            cost_per_1m_requests=(total_cost / requests_per_month * 1_000_000) if requests_per_month > 0 else 0,
            breakdown={
                "request_cost": request_cost,
                "cpu_cost": cpu_cost,
                "memory_cost": memory_cost
            },
            assumptions={
                "requests_per_month": requests_per_month,
                "avg_duration_ms": avg_duration_ms,
                "cpu": cpu,
                "memory_gb": memory_gb,
                "includes_free_tier": True
            }
        )

    def estimate_kubernetes(
        self,
        node_type: str = "n1-standard-2",
        num_nodes: int = 3,
        requests_per_month: int = 1_000_000
    ) -> CostEstimate:
        """Estimate Kubernetes deployment costs."""
        pricing = self.PRICING["kubernetes"]

        node_costs = pricing["node_costs"]
        if node_type not in node_costs:
            node_type = "n1-standard-2"

        hours_per_month = 730

        # Node costs
        node_cost = node_costs[node_type] * hours_per_month * num_nodes

        # Load balancer
        lb_cost = pricing["load_balancer"] * hours_per_month

        # Storage (estimate 100GB)
        storage_cost = 100 * pricing["persistent_disk_gb"]

        total_cost = node_cost + lb_cost + storage_cost

        return CostEstimate(
            platform="Kubernetes",
            monthly_cost_usd=total_cost,
            cost_per_1k_requests=(total_cost / requests_per_month * 1000) if requests_per_month > 0 else 0,
            cost_per_1m_requests=(total_cost / requests_per_month * 1_000_000) if requests_per_month > 0 else 0,
            breakdown={
                "node_cost": node_cost,
                "load_balancer_cost": lb_cost,
                "storage_cost": storage_cost
            },
            assumptions={
                "node_type": node_type,
                "num_nodes": num_nodes,
                "storage_gb": 100,
                "requests_per_month": requests_per_month
            }
        )

    def compare_platforms(
        self,
        requests_per_month: int,
        avg_duration_ms: int,
        memory_gb: float = 1.0
    ) -> List[CostEstimate]:
        """
        Compare costs across all platforms.

        Args:
            requests_per_month: Expected requests per month
            avg_duration_ms: Average request duration
            memory_gb: Memory requirement

        Returns:
            List of cost estimates, sorted by cost
        """
        estimates = []

        # Lambda
        estimates.append(self.estimate_lambda(
            requests_per_month,
            avg_duration_ms,
            int(memory_gb * 1024)
        ))

        # SageMaker
        estimates.append(self.estimate_sagemaker(
            "ml.t2.medium",
            1,
            requests_per_month
        ))

        # Cloud Run
        estimates.append(self.estimate_cloud_run(
            requests_per_month,
            avg_duration_ms,
            1.0,
            memory_gb
        ))

        # Kubernetes
        estimates.append(self.estimate_kubernetes(
            "n1-standard-2",
            3,
            requests_per_month
        ))

        # Sort by cost
        estimates.sort(key=lambda x: x.monthly_cost_usd)

        return estimates

    def recommend_platform(
        self,
        requests_per_month: int,
        avg_duration_ms: int,
        memory_gb: float = 1.0,
        requirements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Recommend best deployment platform.

        Args:
            requests_per_month: Expected requests per month
            avg_duration_ms: Average request duration
            memory_gb: Memory requirement
            requirements: Additional requirements

        Returns:
            Platform recommendation with reasoning
        """
        requirements = requirements or {}

        estimates = self.compare_platforms(
            requests_per_month,
            avg_duration_ms,
            memory_gb
        )

        # Filter based on requirements
        if requirements.get('gpu_required'):
            # Only SageMaker or K8s support GPU easily
            filtered = [e for e in estimates if e.platform in ['AWS SageMaker', 'Kubernetes']]
        else:
            filtered = estimates

        best = filtered[0] if filtered else estimates[0]

        return {
            'recommended_platform': best.platform,
            'estimated_monthly_cost': best.monthly_cost_usd,
            'cost_per_1m_requests': best.cost_per_1m_requests,
            'reasoning': self._generate_reasoning(best, estimates, requirements),
            'all_estimates': estimates
        }

    def _generate_reasoning(
        self,
        recommended: CostEstimate,
        all_estimates: List[CostEstimate],
        requirements: Dict[str, Any]
    ) -> str:
        """Generate reasoning for platform recommendation."""
        reasons = []

        # Cost-based reasoning
        if recommended == all_estimates[0]:
            savings = all_estimates[1].monthly_cost_usd - recommended.monthly_cost_usd
            reasons.append(f"Lowest cost option, saving ${savings:.2f}/month")

        # Traffic pattern
        if recommended.platform == "AWS Lambda":
            reasons.append("Ideal for variable workloads with automatic scaling")
        elif recommended.platform == "AWS SageMaker":
            reasons.append("Best for production ML with managed infrastructure and monitoring")
        elif recommended.platform == "Kubernetes":
            reasons.append("Maximum control and flexibility for complex deployments")

        return "; ".join(reasons)
