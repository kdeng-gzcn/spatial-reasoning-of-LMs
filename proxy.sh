sudo bash ../clash-for-linux/start.sh
source /etc/profile.d/clash.sh
proxy_on
env | grep -E 'http_proxy|https_proxy'