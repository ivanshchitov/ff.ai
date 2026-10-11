#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="${TMPDIR:-/tmp}/solve-challenge-task"
STATE_FILE="$STATE_DIR/recorder.state"
MAX_SECONDS=1800

usage() {
    cat >&2 <<'EOF'
record_screen.sh — запись экрана для демонстрации решения (Linux X11/Wayland, macOS)

  doctor            проверить платформу и доступные рекордеры
  start <basename>  начать запись в <basename>.<ext>; ext выбирает рекордер (.mp4/.mkv)
  stop              корректно остановить запись (SIGINT) и показать размер файла
EOF
    exit 2
}

log() { printf '%s\n' "$*" >&2; }
die() { log "ОШИБКА: $*"; exit 1; }

pid_alive() { kill -0 "$1" 2>/dev/null; }

require_state() {
    [ -f "$STATE_FILE" ] || die "запись не запущена"
    IFS='|' read -r PID WDPID OUTFILE < "$STATE_FILE" || die "повреждённый state-файл: $STATE_FILE"
}

# 2-секундная проба захвата в $STATE_DIR: без разрешения «Запись экрана» macOS
# отдаёт чёрный кадр, а ffmpeg при этом завершается успешно — ловим по размеру PNG.
macos_probe() {
    mkdir -p "$STATE_DIR"
    probe="$STATE_DIR/probe.mp4"; frame="$STATE_DIR/probe.png"
    rm -f "$probe" "$frame"
    ffmpeg -hide_banner -loglevel error -f avfoundation -framerate 30 -pixel_format uyvy422 \
        -i "$1:none" -t 2 -vcodec libx264 -preset ultrafast -pix_fmt yuv420p -y "$probe" \
        2>/dev/null || return 1
    ffmpeg -y -v error -ss 1 -i "$probe" -frames:v 1 "$frame" 2>/dev/null || return 1
    size="$(stat -f %z "$frame" 2>/dev/null || stat -c %s "$frame" 2>/dev/null || echo 0)"
    rm -f "$probe" "$frame"
    [ "$size" -gt 10000 ]
}

cmd_doctor() {
    os="$(uname -s)"
    log "ОС: $os"
    if [ "$os" = "Darwin" ]; then
        log "Сессия: macOS"
        if command -v ffmpeg >/dev/null 2>&1; then
            log "ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"
            idx="$(ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 \
                | sed -n 's/.*\[\([0-9]*\)\] Capture screen.*/\1/p' | head -1 || true)"
            if [ -n "${idx:-}" ]; then
                log "avfoundation: экран найден (устройство $idx)"
                if macos_probe "$idx"; then
                    log "проба 2 сек: кадр не чёрный"
                    log "ВЕРДИКТ: ГОТОВ (ffmpeg avfoundation)"
                else
                    log "ВЕРДИКТ: НЕ ГОТОВ — проба дала чёрный/пустой кадр: у терминала нет разрешения"
                    log "«Запись экрана» (Системные настройки → Конфиденциальность и безопасность → Запись экрана)"
                fi
            else
                log "avfoundation: экран не найден, будет fallback screencapture"
                command -v screencapture >/dev/null 2>&1 \
                    && log "ВЕРДИКТ: ГОТОВ (screencapture -V)" \
                    || log "ВЕРДИКТ: НЕ ГОТОВ (нет ни ffmpeg-экрана, ни screencapture)"
            fi
        elif command -v screencapture >/dev/null 2>&1; then
            log "ВЕРДИКТ: ГОТОВ (screencapture -V, без ffmpeg: без проверки кадра)"
        else
            log "ВЕРДИКТ: НЕ ГОТОВ — нужен ffmpeg (brew install ffmpeg)"
        fi
        return 0
    fi

    if [ -n "${WAYLAND_DISPLAY:-}" ] || [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
        log "Сессия: Wayland (WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-?})"
        if command -v wf-recorder >/dev/null 2>&1; then
            log "ВЕРДИКТ: ГОТОВ (wf-recorder)"
        elif command -v gpu-screen-recorder >/dev/null 2>&1; then
            kms="/usr/bin/gsr-kms-server"
            if [ ! -x "$kms" ]; then
                log "ВЕРДИКТ: НЕ ГОТОВ — есть gpu-screen-recorder, но нет $kms (битый пакет)"
            elif command -v getcap >/dev/null 2>&1 && getcap "$kms" 2>/dev/null | grep -q cap_sys_admin; then
                log "ВЕРДИКТ: ГОТОВ (gpu-screen-recorder, kms cap_sys_admin есть)"
            else
                log "ВЕРДИКТ: НЕ ГОТОВ — gsr-kms-server без cap_sys_admin: захват упадёт с пустым файлом"
                log "Фикс (один раз, нужен sudo): sudo setcap cap_sys_admin+ep $kms"
            fi
        else
            log "ВЕРДИКТ: НЕ ГОТОВ — x11grab на Wayland даст чёрный экран"
            log "Установи: wf-recorder (wlroots) или gpu-screen-recorder (GNOME/KDE Wayland)"
        fi
        return 0
    fi

    log "Сессия: X11 (DISPLAY=${DISPLAY:-unset})"
    if command -v ffmpeg >/dev/null 2>&1; then
        log "ВЕРДИКТ: ГОТОВ (ffmpeg x11grab)"
    else
        log "ВЕРДИКТ: НЕ ГОТОВ — нужен ffmpeg (apt install ffmpeg)"
    fi
}

recorder_for_linux_wayland() {
    if command -v wf-recorder >/dev/null 2>&1; then
        printf 'wf-recorder|mkv\n'
    elif command -v gpu-screen-recorder >/dev/null 2>&1; then
        printf 'gpu-screen-recorder|mp4\n'
    else
        return 1
    fi
}

x11_geometry() {
    if command -v xdpyinfo >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
        geom="$(xdpyinfo 2>/dev/null | sed -n 's/.*dimensions: *\([0-9]*x[0-9]*\).*/\1/p' | head -1)"
        [ -n "${geom:-}" ] && { printf '%s' "$geom"; return 0; }
    fi
    printf '1920x1080'
}

cmd_start() {
    [ $# -ge 1 ] || usage
    base="$1"
    case "$base" in
        */*) outdir="${base%/*}"; mkdir -p "$outdir" ;;
    esac

    if [ -f "$STATE_FILE" ]; then
        IFS='|' read -r PID WDPID OUTFILE < "$STATE_FILE" || true
        if [ -n "${PID:-}" ] && pid_alive "$PID"; then
            die "запись уже идёт: PID $PID → $OUTFILE"
        fi
        rm -f "$STATE_FILE"
    fi

    mkdir -p "$STATE_DIR"
    os="$(uname -s)"
    ext=""
    runner=""

    if [ "$os" = "Darwin" ]; then
        if command -v ffmpeg >/dev/null 2>&1; then
            idx="$(ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 \
                | sed -n 's/.*\[\([0-9]*\)\] Capture screen.*/\1/p' | head -1 || true)"
            if [ -n "${idx:-}" ]; then
                ext="mp4"
                runner="ffmpeg -hide_banner -loglevel error -f avfoundation -framerate 30 -pixel_format uyvy422 -i ${idx}:none -vcodec libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +frag_keyframe+empty_moov"
            else
                die "ffmpeg есть, но экран avfoundation не найден; запусти doctor"
            fi
        else
            command -v screencapture >/dev/null 2>&1 || die "нет ни ffmpeg, ни screencapture"
            ext="mov"
            runner="screencapture -V $MAX_SECONDS"
        fi
    elif [ -n "${WAYLAND_DISPLAY:-}" ] || [ "${XDG_SESSION_TYPE:-}" = "wayland" ]; then
        pick="$(recorder_for_linux_wayland)" || die "нет рекордера для Wayland; запусти doctor"
        rec="${pick%%|*}"; ext="${pick##*|}"
        case "$rec" in
            wf-recorder)          runner="wf-recorder -f" ;;
            gpu-screen-recorder)  runner="gpu-screen-recorder -w screen -f 60 -o" ;;
        esac
    else
        [ -n "${DISPLAY:-}" ] || die "нет ни Wayland, ни DISPLAY — куда писать?"
        command -v ffmpeg >/dev/null 2>&1 || die "нет ffmpeg; запусти doctor"
        ext="mp4"
        geom="$(x11_geometry)"
        runner="ffmpeg -hide_banner -loglevel error -f x11grab -framerate 30 -video_size ${geom} -i ${DISPLAY} -vcodec libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +frag_keyframe+empty_moov"
    fi

    outfile="${base}.${ext}"
    [ "$outfile" != "$base" ] || outfile="${base}-rec.${ext}"

    eval "exec $runner \"\$outfile\"" &
    pid=$!
    sleep 1
    pid_alive "$pid" || die "рекордер упал сразу после старта; запусти doctor и проверь параметры"

    ( sleep "$MAX_SECONDS"; kill -INT "$pid" 2>/dev/null ) &
    wdpid=$!

    printf '%s|%s|%s\n' "$pid" "$wdpid" "$outfile" > "$STATE_FILE"
    log "ЗАПИСЬ ИДЁТ: $outfile"
    log "рекордер: $(printf '%s' "$runner" | cut -d' ' -f1)"
    log "лимит: ${MAX_SECONDS}s; остановка: record_screen.sh stop"
}

cmd_stop() {
    require_state
    kill "$WDPID" 2>/dev/null || true
    pid_alive "$PID" || { rm -f "$STATE_FILE"; die "рекордер уже не работает; файл: $OUTFILE"; }
    kill -INT "$PID" 2>/dev/null || true
    i=0
    while pid_alive "$PID" && [ "$i" -lt 15 ]; do
        sleep 1; i=$((i + 1))
    done
    if pid_alive "$PID"; then
        log "рекордер не завершился за 15s — отправляю SIGKILL (файл может быть битым)"
        kill -9 "$PID" 2>/dev/null || true
    fi
    rm -f "$STATE_FILE"
    if [ -s "$OUTFILE" ]; then
        size="$(du -h "$OUTFILE" | cut -f1)"
        log "ЗАПИСЬ ЗАВЕРШЕНА: $OUTFILE ($size)"
    else
        die "файл записи пуст или отсутствует: $OUTFILE"
    fi
}

case "${1:-}" in
    doctor) shift; cmd_doctor "$@" ;;
    start)  shift; cmd_start "$@" ;;
    stop)   shift; cmd_stop "$@" ;;
    *) usage ;;
esac
