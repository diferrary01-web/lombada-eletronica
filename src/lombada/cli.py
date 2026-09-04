"""Interface de linha de comando."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import __version__
from .config import AppConfig, ConfigError, load_config
from .geometry import image_to_world
from .pipeline import run_all
from .storage import PassageStore

DEFAULT_CONFIG = Path("config/cameras.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        return int(args.handler(args))
    except ConfigError as exc:
        print(f"erro de configuracao: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lombada", description="Lombada eletronica: medicao de velocidade por video"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="valida a configuracao e a calibracao")
    _add_config(check)
    check.set_defaults(handler=cmd_check)

    run = sub.add_parser("run", help="roda o pipeline nas cameras habilitadas")
    _add_config(run)
    run.set_defaults(handler=cmd_run)

    snapshot = sub.add_parser(
        "snapshot", help="salva um quadro da camera para marcar os 4 pontos da base"
    )
    _add_config(snapshot)
    snapshot.add_argument("--camera", required=True)
    snapshot.add_argument("--output", type=Path, default=Path("snapshot.jpg"))
    snapshot.set_defaults(handler=cmd_snapshot)

    project = sub.add_parser(
        "project", help="projeta um pixel no plano da via (confere a calibracao)"
    )
    _add_config(project)
    project.add_argument("--camera", required=True)
    project.add_argument("x", type=float)
    project.add_argument("y", type=float)
    project.set_defaults(handler=cmd_project)

    bench = sub.add_parser(
        "bench", help="avalia o ensemble contra recortes reais com gabarito"
    )
    _add_config(bench)
    bench.add_argument(
        "--images", type=Path, required=True, help="diretorio com os recortes"
    )
    bench.add_argument(
        "--labels", type=Path, help="CSV `arquivo,placa`; sem ele o gabarito e o nome"
    )
    bench.add_argument(
        "--engines", help="lista separada por virgula, sobrepoe o YAML"
    )
    bench.set_defaults(handler=cmd_bench)

    report = sub.add_parser("report", help="resumo e ultimas passagens")
    _add_config(report)
    report.add_argument("--days", type=int, default=1)
    report.add_argument("--limit", type=int, default=20)
    report.add_argument("--camera")
    report.add_argument("--violations", action="store_true")
    report.add_argument("--json", action="store_true")
    report.set_defaults(handler=cmd_report)

    purge = sub.add_parser("purge", help="aplica a retencao configurada")
    _add_config(purge)
    purge.set_defaults(handler=cmd_purge)

    return parser


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)


def cmd_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(f"local: {config.site_name} ({config.timezone})")
    print(f"detector: {config.detector.backend} / {config.detector.model}")
    if config.lpr.enabled:
        motores = ", ".join(
            f"{name} (peso {config.lpr.weight_of(name):.2f})"
            for name in config.lpr.engines
        )
        print(f"lpr: ensemble de {len(config.lpr.engines)} -- {motores}")
        if config.lpr.min_agreement > 0:
            print(f"     concordancia minima: {config.lpr.min_agreement:.2f}")
    else:
        print("lpr: desligado")
    print(
        f"banco: {config.storage.database_path}  "
        f"retencao: {config.storage.retention_days} dias"
    )
    print()
    for cam in config.cameras:
        homography = cam.base.homography()
        corners = [image_to_world(homography, p) for p in cam.base.image_points]
        estado = "ativa" if cam.enabled else "desativada"
        janela = (
            f"{cam.schedule.start:%H:%M}-{cam.schedule.end:%H:%M}"
            if cam.schedule
            else "24h"
        )
        print(f"[{cam.id}] {cam.name} -- {estado}, {janela}")
        print(
            f"    base {cam.base.distance_m:.1f} m, limite {cam.limit_kmh:.0f} km/h,"
            f" captura {cam.capture_fps:.0f} fps"
        )
        print(
            "    cantos reprojetados: "
            + ", ".join(f"({x:+.2f}, {y:+.2f})" for x, y in corners)
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    stop = threading.Event()

    def handle(signum: int, _frame: object) -> None:
        logging.getLogger(__name__).info("sinal %s recebido, encerrando", signum)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle)

    pipelines = run_all(config, stop)
    for pipeline in pipelines:
        print(
            f"{pipeline.camera.id}: {pipeline.measured} passagens medidas, "
            f"{pipeline.violations} acima do limite"
        )
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    import cv2

    config = load_config(args.config)
    camera = config.camera(args.camera)
    source = camera.substream_url or camera.rtsp_url
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            print("nao consegui abrir o fluxo da camera", file=sys.stderr)
            return 1
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        print("nao consegui ler um quadro", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), frame)
    print(f"quadro salvo em {args.output} ({frame.shape[1]}x{frame.shape[0]})")
    print("marque os 4 cantos da base e preencha `base.image_points` na ordem:")
    print("  entrada-esquerda, entrada-direita, saida-direita, saida-esquerda")
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    camera = config.camera(args.camera)
    world_x, world_y = image_to_world(camera.base.homography(), (args.x, args.y))
    print(f"pixel ({args.x:.1f}, {args.y:.1f}) -> X={world_x:+.2f} m, Y={world_y:+.2f} m")
    if not -1.0 <= world_y <= camera.base.distance_m + 1.0:
        print("aviso: o ponto cai fora da base de medicao")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    import dataclasses

    from .bench import evaluate, format_report, load_samples
    from .config import LprConfig
    from .lpr import build_reader

    # A bancada nao precisa de camera configurada: so de motores e imagens.
    if args.config.is_file():
        lpr = load_config(args.config).lpr
    else:
        lpr = LprConfig()
        print(f"(sem {args.config}: usando os padroes de LPR)")

    if args.engines:
        nomes = tuple(e.strip() for e in args.engines.split(",") if e.strip())
        lpr = dataclasses.replace(lpr, engines=nomes, enabled=True)

    samples = load_samples(args.images, args.labels)
    if not samples:
        print(f"nenhuma amostra com gabarito em {args.images}", file=sys.stderr)
        return 1

    print(f"motores: {', '.join(lpr.engines)}")
    print(f"amostras: {len(samples)}\n")
    result = evaluate(build_reader(lpr), samples, _load_image)
    print(format_report(result))
    return 0


def _load_image(path: Path) -> Any:
    import cv2

    return cv2.imread(str(path))


def cmd_report(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    store = PassageStore(config.storage.database_path)
    since = datetime.now() - timedelta(days=args.days)

    rows = store.recent(
        limit=args.limit,
        camera_id=args.camera,
        only_violations=args.violations,
        since=since,
    )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"-- ultimos {args.days} dia(s) --")
    for row in store.stats(since=since):
        taxa = (row["with_plate"] or 0) / row["total"] * 100 if row["total"] else 0.0
        print(
            f"{row['camera_id']}: {row['total']} passagens, "
            f"{row['violations'] or 0} acima do limite, "
            f"media {row['avg_kmh']:.1f} km/h, maxima {row['max_kmh']:.1f} km/h, "
            f"placa lida em {taxa:.0f}%"
        )
    print()
    for row in rows:
        plate = row["plate"].text if row["plate"] else "-"
        marca = "!" if row["is_violation"] else " "
        print(
            f"{marca} {row['captured_at'][:19]}  {row['camera_id']:<12} "
            f"{row['speed_kmh']:6.1f} km/h  {plate}"
        )
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    config: AppConfig = load_config(args.config)
    store = PassageStore(config.storage.database_path)
    removed = store.purge_older_than(
        config.storage.retention_days, config.storage.evidence_dir
    )
    print(
        f"{removed} passagens removidas "
        f"(retencao de {config.storage.retention_days} dias)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
