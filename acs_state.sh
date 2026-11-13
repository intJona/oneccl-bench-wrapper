#!/usr/bin/env bash
# Prints the kernel command line and the ACS control word of every bridge above each Intel B70 (8086:e223).
# Reading ACS needs root: uses `sudo -n` (cached sudo). Without it, falls back to the kernel's last
# "Configured ACS" log line for that port (value suffixed "(klog)"), else "unavailable".
echo "kernel_cmdline=$(sed 's/ crashkernel=[^ ]*//' /proc/cmdline)"
have_sudo=0; sudo -n true 2>/dev/null && have_sudo=1
declare -A klog
while read -r b v; do klog[$b]=$v; done < <(journalctl -k -b --no-pager 2>/dev/null \
  | sed -nE 's/.*(0000:[0-9a-f:.]+): Configured ACS to 0x([0-9a-f]{4}).*/\1 \2/p')
for g in $(lspci -D -n | awk '/8086:e223/ {print $1}'); do
  d=$(readlink -f "/sys/bus/pci/devices/$g")
  echo "gpu=$g path=${d#/sys/devices/}"
  while d=$(dirname "$d"); [[ -e $d/vendor ]]; do
    b=$(basename "$d")
    if (( have_sudo )); then
      v=$(sudo -n setpci -s "$b" ECAP_ACS+6.w 2>/dev/null)
      [[ $v =~ ^[0-9a-f]{4}$ ]] || v=none
    elif [[ -n ${klog[$b]:-} ]]; then
      v="${klog[$b]}(klog)"
    else
      v=unavailable
    fi
    echo "acs $b=$v"
  done
done
