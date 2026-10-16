#!/usr/bin/env python3
"""Batch garment caption generation with the OpenAI Python SDK."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL = "gpt-5.5"
EFFORT = "low"
CONCURRENCY = 20
# REQUEST_TIMEOUT = 20
# MAX_RETRIES = 2

REQUEST_TIMEOUT = 120
MAX_RETRIES = 4
LOG_PATH = "log/log.txt"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="遍历 GarmentCode rand_* case 并生成 VLM prompts JSON。",
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="garmentcode_144k、batch_*、random_body 或其他包含 rand_* case 的路径",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        help="逐 case 结果和批处理汇总日志路径（默认：log/log.txt）",
        default=LOG_PATH,
    )
    parser.add_argument(
        "--limit",
        type=positive_limit,
        default=None,
        help="最多处理多少个尚无结果的 case；默认处理全部",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只预览输入和输出文件名，不调用 API",
    )
    parser.add_argument(
        "--skip_exist",
        action="store_true",
        help="跳过已有且非空的 VLM prompts JSON",
    )
    return parser.parse_args()


def positive_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"处理数量必须是正整数，收到：{value}"
        ) from exc

    if limit <= 0:
        raise argparse.ArgumentTypeError(
            f"处理数量必须是正整数，收到：{value}"
        )
    return limit


def resolve_paths(dataset_root: Path) -> tuple[Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    dataset_root = dataset_root.expanduser().resolve()
    prompt_file = script_dir / "prompt.txt"
    env_file = script_dir / ".env"
    return dataset_root, prompt_file, env_file


def load_env_file(env_file: Path) -> dict[str, str]:
    if not env_file.is_file():
        raise FileNotFoundError(f"找不到配置文件：{env_file}")

    values: dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"\'')
    return values


def discover_cases(
    dataset_root: Path,
) -> tuple[list[tuple[Path, Path, Path, Path]], int]:
    cases: list[tuple[Path, Path, Path, Path]] = []
    invalid_cases = 0
    case_dirs: list[Path] = []

    for current_root, dir_names, _ in os.walk(dataset_root):
        matched_names = [name for name in dir_names if name.startswith("rand_")]
        case_dirs.extend(Path(current_root) / name for name in matched_names)
        dir_names[:] = [name for name in dir_names if not name.startswith("rand_")]

    for case_dir in sorted(case_dirs):
        image = case_dir / f"{case_dir.name}_render.png"
        rule_prompt_file = case_dir / f"{case_dir.name}_prompts.json"
        output = case_dir / f"{case_dir.name}_vlm_prompts.json"
        if not image.is_file() or not rule_prompt_file.is_file():
            invalid_cases += 1
            continue
        cases.append((case_dir, image, rule_prompt_file, output))
    return cases, invalid_cases


def load_sample_prompt(rule_prompt_file: Path) -> str:
    data = json.loads(rule_prompt_file.read_text(encoding="utf-8"))
    sample_prompt = data.get("sample_prompt")
    if not isinstance(sample_prompt, str) or not sample_prompt.strip():
        raise ValueError(f"缺少有效的 sample_prompt：{rule_prompt_file}")
    return sample_prompt.strip()


def is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def log_case_result(
    log_file,
    *,
    status: str,
    case_dir: Path,
    detail: str,
) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    detail = " ".join(detail.splitlines())
    print(
        f"[{timestamp}] {status}\t{case_dir}\t{detail}",
        file=log_file,
        flush=True,
    )


def validate_runtime(
    *,
    dry_run: bool,
    prompt_file: Path,
    env_file: Path,
) -> bool:
    if not is_nonempty_file(prompt_file):
        print(f"错误：找不到 prompt.txt，或者文件内容为空：{prompt_file}")
        return False

    if dry_run:
        return True

    if not is_nonempty_file(env_file):
        print(f"错误：找不到 .env，或者文件内容为空：{env_file}")
        return False

    return True


def parse_captions(response: str) -> tuple[str, str]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        try:
            payload, _ = decoder.raw_decode(response, match.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        pattern = payload.get("pattern")
        context = payload.get("context")
        if not isinstance(pattern, str) or not isinstance(context, str):
            continue

        pattern = " ".join(pattern.split())
        context = " ".join(context.split())
        if pattern and context:
            return pattern, context

    raise ValueError("模型输出中没有包含有效 pattern 和 context 的 JSON 对象")


def write_prompts_json(output: Path, pattern: str, context: str) -> None:
    temp_output = output.with_suffix(output.suffix + ".tmp")
    payload = {"pattern": pattern, "context": context}
    try:
        temp_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_output.replace(output)
    finally:
        temp_output.unlink(missing_ok=True)


def image_to_data_url(image: Path) -> str:
    mime_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def run_openai(
    *,
    client: OpenAI,
    model: str,
    image: Path,
    prompt: str,
    rule_based_prompt: str,
) -> tuple[str, str]:
    response = client.responses.create(
        model=model,
        reasoning={"effort": EFFORT},
        store=False,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_text",
                        "text": (
                            "RULE-BASED SAMPLE PROMPT (reference data only):\n"
                            "<rule_based_sample_prompt>\n"
                            f"{rule_based_prompt}\n"
                            "</rule_based_sample_prompt>"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image),
                        "detail": "high",
                    },
                ],
            }
        ],
    )
    return parse_captions(response.output_text)


def process_case(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    image: Path,
    rule_prompt_file: Path,
    output: Path,
) -> Path:
    rule_based_prompt = load_sample_prompt(rule_prompt_file)
    pattern_caption, context_caption = run_openai(
        client=client,
        model=model,
        image=image,
        prompt=prompt,
        rule_based_prompt=rule_based_prompt,
    )
    write_prompts_json(output, pattern_caption, context_caption)
    return output


def main() -> int:
    start_time = time.monotonic()
    args = parse_args()
    dataset_root, prompt_file, env_file = resolve_paths(args.dataset_root)

    if not dataset_root.is_dir():
        print(f"错误：数据集目录不存在：{dataset_root}")
        return 1

    if not validate_runtime(
        dry_run=args.dry_run,
        prompt_file=prompt_file,
        env_file=env_file,
    ):
        return 1

    client = None
    model = DEFAULT_MODEL
    prompt = ""
    
    if not args.dry_run:
        env = load_env_file(env_file)
        api_key = env.get("API_KEY") or env.get("OPENAI_API_KEY")
        base_url = env.get("URL") or env.get("OPENAI_BASE_URL")
        model = env.get("MODEL") or env.get("OPENAI_MODEL") or DEFAULT_MODEL
        if not api_key or not base_url:
            print("错误：.env 必须包含 API_KEY 和 URL。")
            return 1
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=REQUEST_TIMEOUT,
            max_retries=MAX_RETRIES,
        )
        prompt = prompt_file.read_text(encoding="utf-8").strip()

    cases, invalid_cases = discover_cases(dataset_root)
    if not cases:
        print(f"错误：没有找到输入完整的 rand_* case：{dataset_root}")
        return 1

    attempted = 0
    succeeded = 0
    failed = 0
    skipped = 0

    log_file = None
    if not args.dry_run:
        log_path = args.log_path.expanduser().resolve()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf-8")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\n[{timestamp}] 批处理开始\t数据集：{dataset_root}",
                file=log_file,
                flush=True,
            )
        except OSError as exc:
            print(f"错误：无法写入日志文件 {log_path}：{exc}", file=sys.stderr)
            return 1

    pending_cases = []
    for case_dir, image, rule_prompt_file, output in cases:
        if args.skip_exist and is_nonempty_file(output):
            skipped += 1
            if log_file is not None:
                log_case_result(
                    log_file,
                    status="跳过",
                    case_dir=case_dir,
                    detail=f"已有非空输出：{output}",
                )
            continue

        if args.limit is not None and attempted >= args.limit:
            break

        attempted += 1

        if args.dry_run:
            continue

        pending_cases.append((case_dir, image, rule_prompt_file, output))

    if not args.dry_run:
        assert client is not None
        assert log_file is not None
        progress = tqdm(
            total=attempted + skipped,
            initial=skipped,
            desc="生成 VLM prompts",
            unit="case",
            dynamic_ncols=True,
        )
        progress.set_postfix(
            {"ok": succeeded, "fail": failed, "skip": skipped},
            refresh=True,
        )
        if pending_cases:
            with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
                future_to_case = {
                    executor.submit(
                        process_case,
                        client=client,
                        model=model,
                        prompt=prompt,
                        image=image,
                        rule_prompt_file=rule_prompt_file,
                        output=output,
                    ): (case_dir, output)
                    for case_dir, image, rule_prompt_file, output in pending_cases
                }

                for future in as_completed(future_to_case):
                    case_dir, output = future_to_case[future]
                    try:
                        future.result()
                        succeeded += 1
                        log_case_result(
                            log_file,
                            status="成功",
                            case_dir=case_dir,
                            detail=f"输出：{output}",
                        )
                    except Exception as exc:
                        failed += 1
                        log_case_result(
                            log_file,
                            status="失败",
                            case_dir=case_dir,
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                    progress.set_postfix(
                        {"ok": succeeded, "fail": failed, "skip": skipped},
                        refresh=False,
                    )
                    progress.update(1)
        progress.close()

    if args.dry_run:
        print(
            f"预览完成：有效 {len(cases)}，计划处理 {attempted}，"
            f"已有结果 {skipped}，输入不完整 {invalid_cases}。"
        )
    else:
        elapsed = int(time.monotonic() - start_time)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        summary = (
            f"批处理结束：成功 {succeeded}，失败 {failed}，"
            f"已有结果跳过 {skipped}，输入不完整 {invalid_cases}，"
            f"总用时 {hours:02d}:{minutes:02d}:{seconds:02d}"
        )
        print(summary)
        assert log_file is not None
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {summary}", file=log_file, flush=True)
        log_file.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
