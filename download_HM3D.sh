source .env

mkdir -p "data/habitat-matterport-3d"
cd "data/habitat-matterport-3d"

# mkdir -p "example"
# cd "example"
# mkdir -p "glb"
# tar -xvf hm3d-example-glb-v0.2.tar -C "./glb"
# mkdir -p "habitat"
# tar -xvf hm3d-example-habitat-v0.2.tar -C "./habitat"
# mkdir -p "semantic-annots"
# tar -xvf hm3d-example-semantic-annots-v0.2.tar -C "./semantic-annots"
# mkdir -p "semantic-configs"
# tar -xvf hm3d-example-semantic-configs-v0.2.tar -C "./semantic-configs"
# rm -rf hm3d-example-glb-v0.2.tar
# rm -rf hm3d-example-habitat-v0.2.tar
# rm -rf hm3d-example-semantic-annots-v0.2.tar
# rm -rf hm3d-example-semantic-configs-v0.2.tar
# cd "../"

mkdir -p "val"
cd "val"
wget https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar
wget https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar
wget https://api.matterport.com/resources/habitat/hm3d-val-semantic-configs-v0.2.tar
mkdir -p "glb"
tar -xvf hm3d-val-glb-v0.2.tar -C "./glb"
mkdir -p "habitat"
tar -xvf hm3d-val-habitat-v0.2.tar -C "./habitat"
mkdir -p "semantic-annots"
tar -xvf hm3d-val-semantic-annots-v0.2.tar -C "./semantic-annots"
mkdir -p "semantic-configs"
tar -xvf hm3d-val-semantic-configs-v0.2.tar -C "./semantic-configs"
rm -rf hm3d-val-glb-v0.2.tar
rm -rf hm3d-val-habitat-v0.2.tar
rm -rf hm3d-val-semantic-annots-v0.2.tar
rm -rf hm3d-val-semantic-configs-v0.2.tar
cd "../"

mkdir -p "minival"
cd "minival"
curl 'https://api.matterport.com/resources/habitat/hm3d-minival-glb-v0.2.tar' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6' \
  -b '_cfuvid=VlkspwdupAcpZzSUDpFzUgKlFjZt78.65CC_wxTtnK4-1745460595241-0.0.1.1-604800000; _vwo_uuid_v2=D9F3CA3067B6D458258C16B8C4EA586C7|cfa4268260bf42f175f9e15aefdb9c82; _vwo_uuid=D9F3CA3067B6D458258C16B8C4EA586C7; _vis_opt_s=1%7C; _vis_opt_test_cookie=1; _vis_opt_exp_113_combi=2; _vwo_ds=3%3Aa_0%2Ct_0%3A0%241745460595%3A39.63986515%3A%3A%3A%3A0; ajs_anonymous_id=0987b2a2-7e6e-4c3e-8705-d8246c5d2697; apple_analytics=web-referral; _mkto_trk=id:911-LXO-192&token:_mch-matterport.com-9de25427fe84c6ecb263f2d93980f80; _gcl_au=1.1.1990120376.1745460601; FPAU=1.1.1990120376.1745460601; SESSvl=en; _clck=o25tvp%7C2%7Cfvc%7C1%7C1940; _fbp=fb.1.1745460745932.626242922329607199; cbar_uid=9028444371626; cbar_sess=1; singular_device_id=d8b5b6a6-620c-45dc-bec4-dc4a9f3a7860; cbar_lvt=1745460851; domain_token=e8c41248703441b7a91626c9b382222e; authn_token=0cf9848bdacb4a33add4bd73118db3c0; ajs_user_id=LusEFVuE6hN; ajs_group_id=2TzdKcWopzd; cbar_sess_pv=6; cookie_consent_v3=%7B%22version%22%3A3%2C%22strictlyNecessary%22%3Atrue%2C%22custom%22%3A%7B%22performance%22%3Atrue%2C%22functionality%22%3Atrue%2C%22targeting%22%3Atrue%7D%2C%22usChecked%22%3Atrue%7D; mp_membership_sid=2TzdKcWopzd; intercom-device-id-toxdrc11=08310a9a-f677-423e-8d46-112e012a8dbf; intercom-session-toxdrc11=V1BFTUx6K1dCZXQ0QWZGRWllUkc0UXp5Y3lycEtSMnMvZTVBMVJTZkZsQmV0OVk5eldmTFROeUpLREJKL2Zaa0lQMFVjWmptcXM4ZU9FRGNwVGtvKzdGS0hnZlJKWVY5TUxFakNzdEhrVEk9LS1KcXRNaU13UHdISUd3RmlXalRPZzJRPT0=--8a28b9608168e4293aa080684c20ff5681d1846c; ubvt=v2%7C6dae1783-cb19-45c7-8099-f40ed8adc2d3%7Caefe9710-3ca5-478f-8c0c-0a961abd736f%3Aa%3Asingle%3Asingle; _gid=GA1.2.489470508.1745465479; _ga_66Q3QTTX17=GS1.1.1745465478.1.0.1745465483.0.0.2010832891; _ga_351559429=GS1.1.1745465478.1.0.1745465487.0.0.0; _ga_0T33JN78PY=GS1.1.1745465478.1.0.1745465487.0.0.0; __cf_bm=5ikYi0u3.26Dz.Grdjj_dH2XhofIjTkz7ubxb_ezXm0-1745465935-1.0.1.1-6AJ51.lhWsFwu7acFwVVCYssYG1.CK7EZ3cNQOhJZUjgRRQADCZzmpb0M20MUu_Rvzp59NSHY8HaJJI0C.LvPQiMCJhq2U0HYDVX9iqeqNE; intercom-id-xzbuxr48=d5f34e48-6879-4cc5-8813-0a131f553692; intercom-session-xzbuxr48=; intercom-device-id-xzbuxr48=03b05a71-133e-47f2-827f-cd7d52f072cc; _ga=GA1.1.36520656.1745460597; __q_state_oerwbSnkKEjaiD3g=eyJ1dWlkIjoiM2Q4OTg1YWUtOWVmZi00ZjdmLTg0NTMtZGJiOTdiMjUxOWVmIiwiY29va2llRG9tYWluIjoibWF0dGVycG9ydC5jb20iLCJhY3RpdmVTZXNzaW9uSWQiOm51bGwsIm1lc3NlbmdlckV4cGFuZGVkIjpudWxsLCJwcm9tcHREaXNtaXNzZWQiOnRydWUsImNvbnZlcnNhdGlvbklkIjoiMTY0MDA1MDQwOTkzNzYyNTM2MyIsInNjcmlwdElkIjoiMTYzODkwMTkyMzA2NTAwNDI5MCJ9; _uetsid=3a52994020b111f0ab79c97de2040cfb|1qn50lw|2|fvc|0|1940; _vwo_sn=4422%3A10%3A%3A%3A1; _uetvid=3a5296c020b111f0ba7f91aa22410685|18ql1k2|1745466077409|11|1|bat.bing.com/p/insights/c/h; _clsk=1rxqxi6%7C1745466079403%7C2%7C1%7Ce.clarity.ms%2Fcollect; _ga_W66Y5HELXX=GS1.1.1745460597.1.1.1745466330.0.0.1171499942' \
  -H 'priority: u=0, i' \
  -H 'referer: https://github.com/matterport/habitat-matterport-3dresearch/blob/main/README.md' \
  -H 'sec-ch-ua: "Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: document' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-user: ?1' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
curl 'https://api.matterport.com/resources/habitat/hm3d-minival-habitat-v0.2.tar' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6' \
  -b '_cfuvid=VlkspwdupAcpZzSUDpFzUgKlFjZt78.65CC_wxTtnK4-1745460595241-0.0.1.1-604800000; _vwo_uuid_v2=D9F3CA3067B6D458258C16B8C4EA586C7|cfa4268260bf42f175f9e15aefdb9c82; _vwo_uuid=D9F3CA3067B6D458258C16B8C4EA586C7; _vis_opt_s=1%7C; _vis_opt_test_cookie=1; _vis_opt_exp_113_combi=2; _vwo_ds=3%3Aa_0%2Ct_0%3A0%241745460595%3A39.63986515%3A%3A%3A%3A0; ajs_anonymous_id=0987b2a2-7e6e-4c3e-8705-d8246c5d2697; apple_analytics=web-referral; _mkto_trk=id:911-LXO-192&token:_mch-matterport.com-9de25427fe84c6ecb263f2d93980f80; _gcl_au=1.1.1990120376.1745460601; FPAU=1.1.1990120376.1745460601; SESSvl=en; _clck=o25tvp%7C2%7Cfvc%7C1%7C1940; _fbp=fb.1.1745460745932.626242922329607199; cbar_uid=9028444371626; cbar_sess=1; singular_device_id=d8b5b6a6-620c-45dc-bec4-dc4a9f3a7860; cbar_lvt=1745460851; domain_token=e8c41248703441b7a91626c9b382222e; authn_token=0cf9848bdacb4a33add4bd73118db3c0; ajs_user_id=LusEFVuE6hN; ajs_group_id=2TzdKcWopzd; cbar_sess_pv=6; cookie_consent_v3=%7B%22version%22%3A3%2C%22strictlyNecessary%22%3Atrue%2C%22custom%22%3A%7B%22performance%22%3Atrue%2C%22functionality%22%3Atrue%2C%22targeting%22%3Atrue%7D%2C%22usChecked%22%3Atrue%7D; mp_membership_sid=2TzdKcWopzd; intercom-device-id-toxdrc11=08310a9a-f677-423e-8d46-112e012a8dbf; intercom-session-toxdrc11=V1BFTUx6K1dCZXQ0QWZGRWllUkc0UXp5Y3lycEtSMnMvZTVBMVJTZkZsQmV0OVk5eldmTFROeUpLREJKL2Zaa0lQMFVjWmptcXM4ZU9FRGNwVGtvKzdGS0hnZlJKWVY5TUxFakNzdEhrVEk9LS1KcXRNaU13UHdISUd3RmlXalRPZzJRPT0=--8a28b9608168e4293aa080684c20ff5681d1846c; ubvt=v2%7C6dae1783-cb19-45c7-8099-f40ed8adc2d3%7Caefe9710-3ca5-478f-8c0c-0a961abd736f%3Aa%3Asingle%3Asingle; _gid=GA1.2.489470508.1745465479; _ga_66Q3QTTX17=GS1.1.1745465478.1.0.1745465483.0.0.2010832891; _ga_351559429=GS1.1.1745465478.1.0.1745465487.0.0.0; _ga_0T33JN78PY=GS1.1.1745465478.1.0.1745465487.0.0.0; __cf_bm=5ikYi0u3.26Dz.Grdjj_dH2XhofIjTkz7ubxb_ezXm0-1745465935-1.0.1.1-6AJ51.lhWsFwu7acFwVVCYssYG1.CK7EZ3cNQOhJZUjgRRQADCZzmpb0M20MUu_Rvzp59NSHY8HaJJI0C.LvPQiMCJhq2U0HYDVX9iqeqNE; intercom-id-xzbuxr48=d5f34e48-6879-4cc5-8813-0a131f553692; intercom-session-xzbuxr48=; intercom-device-id-xzbuxr48=03b05a71-133e-47f2-827f-cd7d52f072cc; _ga=GA1.1.36520656.1745460597; __q_state_oerwbSnkKEjaiD3g=eyJ1dWlkIjoiM2Q4OTg1YWUtOWVmZi00ZjdmLTg0NTMtZGJiOTdiMjUxOWVmIiwiY29va2llRG9tYWluIjoibWF0dGVycG9ydC5jb20iLCJhY3RpdmVTZXNzaW9uSWQiOm51bGwsIm1lc3NlbmdlckV4cGFuZGVkIjpudWxsLCJwcm9tcHREaXNtaXNzZWQiOnRydWUsImNvbnZlcnNhdGlvbklkIjoiMTY0MDA1MDQwOTkzNzYyNTM2MyIsInNjcmlwdElkIjoiMTYzODkwMTkyMzA2NTAwNDI5MCJ9; _uetsid=3a52994020b111f0ab79c97de2040cfb|1qn50lw|2|fvc|0|1940; _vwo_sn=4422%3A10%3A%3A%3A1; _uetvid=3a5296c020b111f0ba7f91aa22410685|18ql1k2|1745466077409|11|1|bat.bing.com/p/insights/c/h; _clsk=1rxqxi6%7C1745466079403%7C2%7C1%7Ce.clarity.ms%2Fcollect; _ga_W66Y5HELXX=GS1.1.1745460597.1.1.1745466330.0.0.1171499942' \
  -H 'priority: u=0, i' \
  -H 'referer: https://github.com/matterport/habitat-matterport-3dresearch/blob/main/README.md' \
  -H 'sec-ch-ua: "Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: document' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-user: ?1' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
curl 'https://api.matterport.com/resources/habitat/hm3d-minival-semantic-annots-v0.2.tar' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6' \
  -b '_cfuvid=VlkspwdupAcpZzSUDpFzUgKlFjZt78.65CC_wxTtnK4-1745460595241-0.0.1.1-604800000; _vwo_uuid_v2=D9F3CA3067B6D458258C16B8C4EA586C7|cfa4268260bf42f175f9e15aefdb9c82; _vwo_uuid=D9F3CA3067B6D458258C16B8C4EA586C7; _vis_opt_s=1%7C; _vis_opt_test_cookie=1; _vis_opt_exp_113_combi=2; _vwo_ds=3%3Aa_0%2Ct_0%3A0%241745460595%3A39.63986515%3A%3A%3A%3A0; ajs_anonymous_id=0987b2a2-7e6e-4c3e-8705-d8246c5d2697; apple_analytics=web-referral; _mkto_trk=id:911-LXO-192&token:_mch-matterport.com-9de25427fe84c6ecb263f2d93980f80; _gcl_au=1.1.1990120376.1745460601; FPAU=1.1.1990120376.1745460601; SESSvl=en; _clck=o25tvp%7C2%7Cfvc%7C1%7C1940; _fbp=fb.1.1745460745932.626242922329607199; cbar_uid=9028444371626; cbar_sess=1; singular_device_id=d8b5b6a6-620c-45dc-bec4-dc4a9f3a7860; cbar_lvt=1745460851; domain_token=e8c41248703441b7a91626c9b382222e; authn_token=0cf9848bdacb4a33add4bd73118db3c0; ajs_user_id=LusEFVuE6hN; ajs_group_id=2TzdKcWopzd; cbar_sess_pv=6; cookie_consent_v3=%7B%22version%22%3A3%2C%22strictlyNecessary%22%3Atrue%2C%22custom%22%3A%7B%22performance%22%3Atrue%2C%22functionality%22%3Atrue%2C%22targeting%22%3Atrue%7D%2C%22usChecked%22%3Atrue%7D; mp_membership_sid=2TzdKcWopzd; intercom-device-id-toxdrc11=08310a9a-f677-423e-8d46-112e012a8dbf; intercom-session-toxdrc11=V1BFTUx6K1dCZXQ0QWZGRWllUkc0UXp5Y3lycEtSMnMvZTVBMVJTZkZsQmV0OVk5eldmTFROeUpLREJKL2Zaa0lQMFVjWmptcXM4ZU9FRGNwVGtvKzdGS0hnZlJKWVY5TUxFakNzdEhrVEk9LS1KcXRNaU13UHdISUd3RmlXalRPZzJRPT0=--8a28b9608168e4293aa080684c20ff5681d1846c; ubvt=v2%7C6dae1783-cb19-45c7-8099-f40ed8adc2d3%7Caefe9710-3ca5-478f-8c0c-0a961abd736f%3Aa%3Asingle%3Asingle; _gid=GA1.2.489470508.1745465479; _ga_66Q3QTTX17=GS1.1.1745465478.1.0.1745465483.0.0.2010832891; _ga_351559429=GS1.1.1745465478.1.0.1745465487.0.0.0; _ga_0T33JN78PY=GS1.1.1745465478.1.0.1745465487.0.0.0; __cf_bm=5ikYi0u3.26Dz.Grdjj_dH2XhofIjTkz7ubxb_ezXm0-1745465935-1.0.1.1-6AJ51.lhWsFwu7acFwVVCYssYG1.CK7EZ3cNQOhJZUjgRRQADCZzmpb0M20MUu_Rvzp59NSHY8HaJJI0C.LvPQiMCJhq2U0HYDVX9iqeqNE; intercom-id-xzbuxr48=d5f34e48-6879-4cc5-8813-0a131f553692; intercom-session-xzbuxr48=; intercom-device-id-xzbuxr48=03b05a71-133e-47f2-827f-cd7d52f072cc; _ga=GA1.1.36520656.1745460597; __q_state_oerwbSnkKEjaiD3g=eyJ1dWlkIjoiM2Q4OTg1YWUtOWVmZi00ZjdmLTg0NTMtZGJiOTdiMjUxOWVmIiwiY29va2llRG9tYWluIjoibWF0dGVycG9ydC5jb20iLCJhY3RpdmVTZXNzaW9uSWQiOm51bGwsIm1lc3NlbmdlckV4cGFuZGVkIjpudWxsLCJwcm9tcHREaXNtaXNzZWQiOnRydWUsImNvbnZlcnNhdGlvbklkIjoiMTY0MDA1MDQwOTkzNzYyNTM2MyIsInNjcmlwdElkIjoiMTYzODkwMTkyMzA2NTAwNDI5MCJ9; _uetsid=3a52994020b111f0ab79c97de2040cfb|1qn50lw|2|fvc|0|1940; _vwo_sn=4422%3A10%3A%3A%3A1; _uetvid=3a5296c020b111f0ba7f91aa22410685|18ql1k2|1745466077409|11|1|bat.bing.com/p/insights/c/h; _clsk=1rxqxi6%7C1745466079403%7C2%7C1%7Ce.clarity.ms%2Fcollect; _ga_W66Y5HELXX=GS1.1.1745460597.1.1.1745466330.0.0.1171499942' \
  -H 'priority: u=0, i' \
  -H 'referer: https://github.com/matterport/habitat-matterport-3dresearch/blob/main/README.md' \
  -H 'sec-ch-ua: "Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: document' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-user: ?1' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
curl 'https://api.matterport.com/resources/habitat/hm3d-minival-semantic-configs-v0.2.tar' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7' \
  -H 'accept-language: en-GB,en-US;q=0.9,en;q=0.8,zh-CN;q=0.7,zh;q=0.6' \
  -b '_cfuvid=VlkspwdupAcpZzSUDpFzUgKlFjZt78.65CC_wxTtnK4-1745460595241-0.0.1.1-604800000; _vwo_uuid_v2=D9F3CA3067B6D458258C16B8C4EA586C7|cfa4268260bf42f175f9e15aefdb9c82; _vwo_uuid=D9F3CA3067B6D458258C16B8C4EA586C7; _vis_opt_s=1%7C; _vis_opt_test_cookie=1; _vis_opt_exp_113_combi=2; _vwo_ds=3%3Aa_0%2Ct_0%3A0%241745460595%3A39.63986515%3A%3A%3A%3A0; ajs_anonymous_id=0987b2a2-7e6e-4c3e-8705-d8246c5d2697; apple_analytics=web-referral; _mkto_trk=id:911-LXO-192&token:_mch-matterport.com-9de25427fe84c6ecb263f2d93980f80; _gcl_au=1.1.1990120376.1745460601; FPAU=1.1.1990120376.1745460601; SESSvl=en; _clck=o25tvp%7C2%7Cfvc%7C1%7C1940; _fbp=fb.1.1745460745932.626242922329607199; cbar_uid=9028444371626; cbar_sess=1; singular_device_id=d8b5b6a6-620c-45dc-bec4-dc4a9f3a7860; cbar_lvt=1745460851; domain_token=e8c41248703441b7a91626c9b382222e; authn_token=0cf9848bdacb4a33add4bd73118db3c0; ajs_user_id=LusEFVuE6hN; ajs_group_id=2TzdKcWopzd; cbar_sess_pv=6; cookie_consent_v3=%7B%22version%22%3A3%2C%22strictlyNecessary%22%3Atrue%2C%22custom%22%3A%7B%22performance%22%3Atrue%2C%22functionality%22%3Atrue%2C%22targeting%22%3Atrue%7D%2C%22usChecked%22%3Atrue%7D; mp_membership_sid=2TzdKcWopzd; intercom-device-id-toxdrc11=08310a9a-f677-423e-8d46-112e012a8dbf; intercom-session-toxdrc11=V1BFTUx6K1dCZXQ0QWZGRWllUkc0UXp5Y3lycEtSMnMvZTVBMVJTZkZsQmV0OVk5eldmTFROeUpLREJKL2Zaa0lQMFVjWmptcXM4ZU9FRGNwVGtvKzdGS0hnZlJKWVY5TUxFakNzdEhrVEk9LS1KcXRNaU13UHdISUd3RmlXalRPZzJRPT0=--8a28b9608168e4293aa080684c20ff5681d1846c; ubvt=v2%7C6dae1783-cb19-45c7-8099-f40ed8adc2d3%7Caefe9710-3ca5-478f-8c0c-0a961abd736f%3Aa%3Asingle%3Asingle; _gid=GA1.2.489470508.1745465479; _ga_66Q3QTTX17=GS1.1.1745465478.1.0.1745465483.0.0.2010832891; _ga_351559429=GS1.1.1745465478.1.0.1745465487.0.0.0; _ga_0T33JN78PY=GS1.1.1745465478.1.0.1745465487.0.0.0; __cf_bm=5ikYi0u3.26Dz.Grdjj_dH2XhofIjTkz7ubxb_ezXm0-1745465935-1.0.1.1-6AJ51.lhWsFwu7acFwVVCYssYG1.CK7EZ3cNQOhJZUjgRRQADCZzmpb0M20MUu_Rvzp59NSHY8HaJJI0C.LvPQiMCJhq2U0HYDVX9iqeqNE; intercom-id-xzbuxr48=d5f34e48-6879-4cc5-8813-0a131f553692; intercom-session-xzbuxr48=; intercom-device-id-xzbuxr48=03b05a71-133e-47f2-827f-cd7d52f072cc; _ga=GA1.1.36520656.1745460597; __q_state_oerwbSnkKEjaiD3g=eyJ1dWlkIjoiM2Q4OTg1YWUtOWVmZi00ZjdmLTg0NTMtZGJiOTdiMjUxOWVmIiwiY29va2llRG9tYWluIjoibWF0dGVycG9ydC5jb20iLCJhY3RpdmVTZXNzaW9uSWQiOm51bGwsIm1lc3NlbmdlckV4cGFuZGVkIjpudWxsLCJwcm9tcHREaXNtaXNzZWQiOnRydWUsImNvbnZlcnNhdGlvbklkIjoiMTY0MDA1MDQwOTkzNzYyNTM2MyIsInNjcmlwdElkIjoiMTYzODkwMTkyMzA2NTAwNDI5MCJ9; _uetsid=3a52994020b111f0ab79c97de2040cfb|1qn50lw|2|fvc|0|1940; _vwo_sn=4422%3A10%3A%3A%3A1; _uetvid=3a5296c020b111f0ba7f91aa22410685|18ql1k2|1745466077409|11|1|bat.bing.com/p/insights/c/h; _clsk=1rxqxi6%7C1745466079403%7C2%7C1%7Ce.clarity.ms%2Fcollect; _ga_W66Y5HELXX=GS1.1.1745460597.1.1.1745466330.0.0.1171499942' \
  -H 'priority: u=0, i' \
  -H 'referer: https://github.com/matterport/habitat-matterport-3dresearch/blob/main/README.md' \
  -H 'sec-ch-ua: "Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: document' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-user: ?1' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
mkdir -p "glb"
tar -xvf hm3d-minival-glb-v0.2.tar -C "./glb"
mkdir -p "habitat"
tar -xvf hm3d-minival-habitat-v0.2.tar -C "./habitat"
mkdir -p "semantic-annots"
tar -xvf hm3d-minival-semantic-annots-v0.2.tar -C "./semantic-annots"
mkdir -p "semantic-configs"
tar -xvf hm3d-minival-semantic-configs-v0.2.tar -C "./semantic-configs"
rm -rf hm3d-minival-glb-v0.2.tar
rm -rf hm3d-minival-habitat-v0.2.tar
rm -rf hm3d-minival-semantic-annots-v0.2.tar
rm -rf hm3d-minival-semantic-configs-v0.2.tar
cd "../"
