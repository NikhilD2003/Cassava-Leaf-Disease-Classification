import sys, time, torch, timm
name, s, bs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t0 = time.time()
m = timm.create_model(name, pretrained=True, num_classes=5).cuda().to(memory_format=torch.channels_last)
print("loaded", round(time.time() - t0), "s", flush=True)
opt = torch.optim.AdamW(m.parameters(), 1e-4); sc = torch.amp.GradScaler()
x = torch.randn(bs, 3, s, s).cuda().to(memory_format=torch.channels_last); y = torch.randint(0, 5, (bs,)).cuda()
for i in range(23):
    if i == 3: torch.cuda.synchronize(); t = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        l = torch.nn.functional.cross_entropy(m(x), y)
    opt.zero_grad(); sc.scale(l).backward(); sc.step(opt); sc.update()
torch.cuda.synchronize()
print(name, s, bs, "img/s", round(20 * bs / (time.time() - t)), "memGB", round(torch.cuda.max_memory_allocated() / 2**30, 1), flush=True)
