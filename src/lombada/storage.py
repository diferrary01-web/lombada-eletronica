"""Persistencia das passagens em SQLite.

SQLite e suficiente e proposital: uma lombada gera na ordem de milhares de
registros por noite, o arquivo unico simplifica backup e a operacao nao tem
DBA. WAL ligado para que a leitura do relatorio nao trave a gravacao.

A retencao nao e enfeite. Placa e dado pessoal; guardar passagem sem prazo
transforma um medidor de velocidade num historico de deslocamento das mesmas
pessoas. `purge_older_than` apaga o registro E a evidencia em disco.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .models import Passage, PlateRead

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS passages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT    NOT NULL,
    track_id      INTEGER NOT NULL,
    captured_at   TEXT    NOT NULL,
    speed_kmh     REAL    NOT NULL,
    considered_kmh REAL   NOT NULL,
    limit_kmh     REAL    NOT NULL,
    is_violation  INTEGER NOT NULL,
    label         TEXT    NOT NULL,
    plate_text    TEXT,
    plate_conf    REAL,
    quality_json  TEXT    NOT NULL DEFAULT '{}',
    evidence_json TEXT    NOT NULL DEFAULT '{}',
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_passages_captured
    ON passages (captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_passages_camera_captured
    ON passages (camera_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_passages_plate
    ON passages (plate_text) WHERE plate_text IS NOT NULL;
"""


class PassageStore:
    """Acesso as passagens gravadas."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def save(self, passage: Passage) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO passages (
                    camera_id, track_id, captured_at, speed_kmh, considered_kmh,
                    limit_kmh, is_violation, label, plate_text, plate_conf,
                    quality_json, evidence_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    passage.camera_id,
                    passage.track_id,
                    passage.captured_at.isoformat(),
                    passage.speed_kmh,
                    passage.considered_kmh,
                    passage.limit_kmh,
                    int(passage.is_violation),
                    passage.label,
                    passage.plate.text if passage.plate else None,
                    passage.plate.confidence if passage.plate else None,
                    json.dumps(passage.quality, ensure_ascii=False),
                    json.dumps(passage.evidence, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid or 0)

    def recent(
        self,
        *,
        limit: int = 100,
        camera_id: str | None = None,
        only_violations: bool = False,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if only_violations:
            clauses.append("is_violation = 1")
        if since:
            clauses.append("captured_at >= ?")
            params.append(since.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM passages {where} ORDER BY captured_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def stats(self, *, since: datetime | None = None) -> list[dict[str, Any]]:
        """Resumo por camera: total, infracoes, velocidade media e maxima."""
        where = "WHERE captured_at >= ?" if since else ""
        params: list[Any] = [since.isoformat()] if since else []
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT camera_id,
                       COUNT(*)                        AS total,
                       SUM(is_violation)               AS violations,
                       AVG(speed_kmh)                  AS avg_kmh,
                       MAX(speed_kmh)                  AS max_kmh,
                       SUM(plate_text IS NOT NULL)     AS with_plate
                FROM passages {where}
                GROUP BY camera_id ORDER BY camera_id
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def purge_older_than(self, days: int, evidence_dir: Path | None = None) -> int:
        """Apaga registros e evidencias acima do prazo de retencao."""
        if days <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            doomed = conn.execute(
                "SELECT evidence_json FROM passages WHERE captured_at < ?", (cutoff,)
            ).fetchall()
            removed = conn.execute(
                "DELETE FROM passages WHERE captured_at < ?", (cutoff,)
            ).rowcount

        if evidence_dir is not None:
            for row in doomed:
                for value in json.loads(row["evidence_json"] or "{}").values():
                    _unlink_quietly(evidence_dir / str(value))

        if removed:
            logger.info("retencao: %d passagens removidas (> %d dias)", removed, days)
        return int(removed)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["quality"] = json.loads(data.pop("quality_json") or "{}")
    data["evidence"] = json.loads(data.pop("evidence_json") or "{}")
    data["is_violation"] = bool(data["is_violation"])
    plate_text = data.pop("plate_text")
    plate_conf = data.pop("plate_conf")
    data["plate"] = (
        PlateRead(text=plate_text, confidence=plate_conf or 0.0) if plate_text else None
    )
    return data


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("nao consegui apagar evidencia %s: %s", path, exc)
