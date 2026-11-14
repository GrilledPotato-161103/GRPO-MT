# Optional Bash completion: source scripts/main/completion.bash
_grpo_main_complete() {
    local previous="${COMP_WORDS[COMP_CWORD-1]}"
    case "$previous" in
        --stage) COMPREPLY=( $(compgen -W 'pipeline sft rl' -- "${COMP_WORDS[COMP_CWORD]}") ) ;;
        --split) COMPREPLY=( $(compgen -W 'val test' -- "${COMP_WORDS[COMP_CWORD]}") ) ;;
        --config|--checkpoint|--resume|--output) COMPREPLY=( $(compgen -f -- "${COMP_WORDS[COMP_CWORD]}") ) ;;
        *) COMPREPLY=( $(compgen -W '--help --version --config --stage --split --checkpoint --resume --output --trust-checkpoint --set --dry-run' -- "${COMP_WORDS[COMP_CWORD]}") ) ;;
    esac
}
complete -F _grpo_main_complete train.sh eval.sh scripts/main/train.sh scripts/main/eval.sh
