"""
ModelForge AI - Enterprise CLI Utility
Command-line tool for managing projects, datasets, models, training runs, deployments, and drift.
"""

import sys
import os
import json
import argparse
from modelforge.client import ModelForgeClient, ModelForgeError


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelforge",
        description="ModelForge AI — Enterprise ML Lifecycle & Model Operations CLI",
    )
    parser.add_argument("--api-key", help="ModelForge API Key (or env MODELFORGE_API_KEY)")
    parser.add_argument("--base-url", help="API Gateway URL (default: http://localhost:8000/api/v1)")

    subparsers = parser.add_subparsers(dest="command", help="Available CLI commands")

    # modelforge projects list
    proj_parser = subparsers.add_parser("projects", help="Manage ML Projects")
    proj_sub = proj_parser.add_subparsers(dest="subcommand")
    proj_sub.add_parser("list", help="List all projects")
    
    proj_create = proj_sub.add_parser("create", help="Create project")
    proj_create.add_argument("--name", required=True, help="Project name")
    proj_create.add_argument("--type", default="classification", choices=["classification", "regression", "clustering", "deep_learning"])
    proj_create.add_argument("--description", help="Project description")

    # modelforge datasets upload
    data_parser = subparsers.add_parser("datasets", help="Manage Datasets")
    data_sub = data_parser.add_subparsers(dest="subcommand")
    data_upload = data_sub.add_parser("upload", help="Upload tabular dataset")
    data_upload.add_argument("--project-id", required=True, help="Project UUID")
    data_upload.add_argument("--name", required=True, help="Dataset name")
    data_upload.add_argument("--file", required=True, help="Path to CSV/Parquet file")
    data_upload.add_argument("--target", help="Target label column name")

    # modelforge predict
    pred_parser = subparsers.add_parser("predict", help="Execute real-time prediction")
    pred_parser.add_argument("--endpoint", required=True, help="Deployment endpoint path")
    pred_parser.add_argument("--data", required=True, help="JSON feature payload string or file path")

    # modelforge canary
    canary_parser = subparsers.add_parser("canary", help="Adjust Canary traffic percentage")
    canary_parser.add_argument("--deployment-id", required=True, help="Deployment UUID")
    canary_parser.add_argument("--percentage", type=float, required=True, help="Canary percentage (0-100)")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = ModelForgeClient(api_key=args.api_key, base_url=args.base_url)

    try:
        if args.command == "projects":
            if args.subcommand == "list":
                projects = client.projects.list()
                print(f"Total Projects: {len(projects)}")
                for p in projects:
                    print(f" - [{p.id}] {p.name} ({p.problem_type})")
            elif args.subcommand == "create":
                p = client.projects.create(name=args.name, problem_type=args.type, description=args.description)
                print(f"Created Project: {p.id} ({p.name})")

        elif args.command == "datasets":
            if args.subcommand == "upload":
                print(f"Uploading {args.file} to project {args.project_id}...")
                ds = client.datasets.upload(
                    project_id=args.project_id,
                    name=args.name,
                    file_path_or_buffer=args.file,
                    target_column=args.target,
                )
                print(f"Dataset Ingested & Profiled! ID: {ds.id}")

        elif args.command == "predict":
            if os.path.exists(args.data):
                with open(args.data, "r") as f:
                    features = json.load(f)
            else:
                features = json.loads(args.data)

            res = client.predictions.predict(endpoint_path=args.endpoint, features=features)
            print(json.dumps(res, indent=2))

        elif args.command == "canary":
            dep = client.deployments.get(args.deployment_id) if hasattr(client.deployments, "get") else None
            res = client.request(
                "POST",
                f"/deployments/{args.deployment_id}/canary",
                json_data={"canary_stage_percentage": args.percentage},
            )
            print(f"Updated Canary traffic to {args.percentage}%")

    except ModelForgeError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
