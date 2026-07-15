# jj 阶梯-章鱼（Ladder）工作流 (JjOctopus.md)

一句话：**双轨制 —— 一个 PR 有两条轨**。代码轨 `<pr>` 是纯代码链（基座 `armbase()`，随时可推上游）；
文档轨 `hack-<pr>` 是**二叉阶梯**：第 k 个节点 $H_k$ 就是 merge($H_{k-1}$, $P_k$)，只携带第 k 步的
`private` 文档改动。两者在图上天然配对，所以改文档、解冲突、单步调试都能一键回到"代码 + 本地层 +
对应文档"齐全的工作区。

```text
armbase() ──► P1 ─────────► P2 ─────────► P3       书签: <pr>      纯代码轨（可 push）
                \            \             \
localbase() ──► H1 ────────► H2 ─────────► H3      书签: hack-<pr> 文档轨（每一步一个 merge）
                                             \
                                              ──► @   全部 hack-<pr> 的 merge
```

- **纯二叉**：每个 $H_k$ 只有 2 个父（$H_{k-1}$ 与 $P_k$，$H_0$ = `localbase()`），没有 `mega-<pr>`
  合流节点，也没有"本地层旁路"这条第 3 边（`jj new` **不会**消除冗余父，见 §6）。
- **一对一配对**：`jj mgo <P_k>` 落到 $H_k$，`jj mback` 把改动分回 $P_k$ 与 $H_k$，不需要人肉记
  "哪个 hack 提交对应哪个代码提交"。
- **代码轨永远干净**：$P_k$ 的祖先里没有 `localbase()`、没有 `H_k`，`jj git push -b <pr>` 只推代码链。

---

## 1. 安装

配置片段：`AGENTS/jj-octopus.toml`（canonical 副本，仓库里只有这一份，不要另存）。

**片段里有本仓库取值，不是仓库无关的**：两个旋钮（`armbase()`、`localbase()`）和 `private-commits`
里的书签名都是 triton-ascend 的取值。原样装到别的仓库会直接坏掉，实测两种失败形态：

| 未改的旋钮 | 失败点 |
| --- | --- |
| `armbase()` = `'main-dev@upstream'` | `jj mnew` 建代码轨时报 ``Error: Revision `main-dev@upstream` doesn't exist`` |
| `localbase()` = `'bookmarks(exact:"debug")'` | 没有 `debug` 书签时解析为空集，`jj mnew` 报 `Error: No revisions found to use as parent` |

所以默认**按仓库安装**（`trunk()` 也已经在这个文件里）。注意**不能整段 append**：TOML 不允许同一个
`[table]` 头出现两次，而本仓库的 repo config 已经有 `[revset-aliases]`（`trunk()` 在那里），append 之后 jj
会拒绝加载配置（`TOML parse error ... duplicate key: revset-aliases`）。要把片段的键**并进**已有表：

```bash
# 合并脚本：已有表就插键，没有的表整段追加（幂等，可重复跑）
python3 - "$(jj config path --repo)" AGENTS/jj-octopus.toml <<'MERGE'
import sys, pathlib, re
rc, frag = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).read_text()
sections, cur = [], None
for line in frag.split('\n'):
    m = re.match(r'^\[([^\]]+)\]$', line)
    if m: cur = [m.group(1), []]; sections.append(cur)
    elif cur is not None: cur[1].append(line)
text = rc.read_text()
# 先摘掉上一次插入的块，再按表插入，所以重复运行不会产生重复键
text = re.sub(r'^# >>> jj-octopus fragment.*?\n(?:.*\n)*?^# <<< jj-octopus fragment.*?\n', '', text, flags=re.M)
lines = text.rstrip('\n').split('\n')
def table_end(lines, name):
    for i, line in enumerate(lines):
        if line.strip() == '[%s]' % name:
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith('['):
                    end = j; break
            while end > i + 1 and not lines[end - 1].strip(): end -= 1
            return end
    return None
for name, body in sections:
    block = ['# >>> jj-octopus fragment (%s)' % name] + [l for l in body] + ['# <<< jj-octopus fragment (%s)' % name]
    while len(block) > 2 and not block[-2].strip(): del block[-2]
    at = table_end(lines, name)
    lines = (lines + ['', '[%s]' % name]) + block if at is None else lines[:at] + block + lines[at:]
rc.write_text('\n'.join(lines) + '\n')
MERGE

# 自检（报 Value not found 或 TOML 错误 = 没合并对 / 路径不对）
jj config get aliases.mnew
```

`.jj/repo/config.toml` 在 0.46 **不会被读取**；`jj config path --repo` 才是真实路径。

要全局安装（`~/.config/jj/config.toml`，所有仓库可用），先把片段改成仓库无关形式——`armbase()` 写
`'trunk()'`，`localbase()` 指向一个**本地**书签（`mland` 靠书签名前移它，`main@origin` 这类远端书签不行），
`private-commits` 换成自己的本地层书签名——再在每个仓库的 repo config 里覆盖同名键（后加载的配置层胜出，
实测覆盖 `"armbase()"` 生效）：

```toml
[revset-aliases]
"armbase()" = "main-dev@upstream"
"localbase()" = 'bookmarks(exact:"debug")'
```

三个旋钮：

| 旋钮 | 说明 |
| --- | --- |
| `trunk()` | PR 的**合入目标分支**。上游 `main` 作目标时写 `"main@origin"`（远端书签，`jj git fetch` 会推动它）；<br>本仓库沿用 `~/.config/jj/repos/<id>/config.toml` 里的 `"localhack@origin"`。 |
| `armbase()` | **代码轨基座**。`'trunk()'` = 代码链跟着本地合入目标走；`'main-dev@upstream'` = 代码链直接基于上游 main-dev，可原样推上游评审（本仓库当前取值）。 |
| `localbase()` | **本地层尖端**：构建补丁、本地 hack、沉淀文档都在它上面，阶梯的左根。必须是**本地书签**（`mland` 靠书签名前移它）。本仓库是 `'bookmarks(exact:"debug")'`（`localhack` 的 hacking 子提交）。 |

---

## 2. 命令

| 命令 | 语义 |
| --- | --- |
| `jj mnew <pr> ["描述"]` | 开一对轨：$P_1$ = 在 `armbase()` 上建的空提交（书签 `<pr>`）；$H_1$ = `jj new localbase() <pr>` 的二叉 merge（书签 `hack-<pr>`）；`mmega` 把 `@` 摆到全部 `hack-*` 之上。要求 `@` 干净。 |
| `jj mput <pr> [--absorb]` | `@` 的改动分流：`~private` → `<pr>` 尖端，其余 → `hack-<pr>` 尖端。`--absorb` 先用 `jj absorb --into '(::<pr>) ~ (::armbase())'` 把改动下沉到引入该行的那个代码提交；`@` 里没有非 private 改动、或那些行不属于该 PR 的任何提交时 absorb 会非零退出，此时 `mput` **只提示不中止**，按"尖端 squash"处理。 |
| `jj mcommit <pr> "信息"` | 追加一级阶梯：`jj split -r @ -o <pr> '~private'` 抽出 $P_{k+1}$，再建 $H_{k+1}$ = merge($H_k$, $P_{k+1}$)，本步文档落进 $H_{k+1}$，`@` 换成新的空提交。 |
| `jj mgo <P_k 或 H_k>` | 解冲突/单步调试：落到配对的阶梯节点 $H_k$ 上（工作区同时有 $P_1..P_k$ 的代码、`localbase()` 的编译补丁与本地 hack、$H_1..H_k$ 的文档）。要求 `@` 干净。 |
| `jj mback [--hack-only]` | 从 `mgo` 状态返回：`~private` 回 $P_k$，其余回 $H_k$，再 `mmega` 把 `@` 摆回全部 `hack-*` 之上。 |
| `jj mland <pr>` | PR 合入后：`jj git fetch`；把本地层放到最新 `trunk()`；**按阶梯节点逐级**沉淀文档（每级一个 `docs(<pr>): <该步信息>` 提交）；卸下两条轨；重建其余节点。永不改写本地层。若阶梯 rebase 到新本地层时在 `private` 文件上冲突，命令**中止且不清算**（`localbase` 不前移、`hack-<pr>` 不删），先 `jj mgo <P_k>` 解冲突再重跑。 |
| `jj mmega` | 内部命令，日常不用。把每条阶梯的根 $H_1$ 对齐到最新 `localbase()`（`rebase -s H1 -o localbase() -o P1`，只要 2 个父），删掉遗留的 `mega-*` 书签，再把 `@` 合并到全部 `hack-*` 之上。 |

日常三步（`auth` 与 `pay` 两个并行需求）：

```bash
jj mnew auth "feat: auth skeleton"     # 一个 PR 一条代码轨 + 一条文档轨
# ……写代码、写 AGENTS/auth.md ……
jj mput auth                            # 自动分流：代码 -> auth，文档 -> hack-auth
jj mcommit auth "feat: auth token refresh"   # 追加一级阶梯（代码 + 对应文档配对）
jj git push -b auth                     # 只推代码轨
jj mland auth                           # 合入上游后：文档逐级沉淀进本地层
```

### 解冲突 / 单步调试

```bash
jj mgo <P_2 的 change id>    # 跳到 H_2：代码(1..2) + 本地层 + 文档(1..2) 都在
# ……改文件、编译、跑 UT……
jj mback                     # 代码回 P_2，文档/合并修正回 H_2，@ 回到顶层
```

`mgo` 也接受 $H_k$ 本身；`mback --hack-only` 表示"这次冲突只按本地层口径解决，不动代码提交"。

---

## 3. 双轨制的边界

| 位置 | 内容 | 约束 |
| --- | --- | --- |
| 代码轨 `<pr>` | 第 k 步的代码 | 不得含 `private` 文件（`AGENTS.md`/`AGENTS/**`/`CLAUDE.md`）；祖先里不含 `localbase()` |
| 文档轨 `hack-<pr>` | 第 k 步的文档 | 每个节点只携带本步 `private` 改动；节点 = merge($H_{k-1}$, $P_k$) |
| `localbase()` | 本地层：构建补丁、本地 hack、已着陆 PR 的文档 | 只追加不改写（`mland` 一律在尖端新建提交） |

`private-commits = 'localbase():: | bookmarks(exact:"localhack") | bookmarks(glob:"mega-*")'` 只是
`jj git push` 的闸门（实测它**不阻止** rewrite：对"private"提交做 `squash`/`abandon`/`rebase` 都照常成功），
所以把整条本地层后代都算进来是安全的；代码轨不在这条链上，推得出去。

---

## 4. 本仓库（triton-ascend）落地注意

- **旋钮取值的存放位置**：`armbase()` = `main-dev@upstream`（代码轨直接基于上游，推上去就是可评审的 PR）、
  `localbase()` = `bookmarks(exact:"debug")`（`localhack` 之上放本地 hack 的那个子提交）写在**片段里**，
  所以片段按仓库安装；`trunk()` = `localhack@origin`（PR 合入目标）写在 repo config 里。
- **`debug` 是本地层的一部分**：阶梯节点的左根是 `debug`，所以 `mgo` 进入任意一步时，`debug` 里的本地 hack
  （`.gitignore` / `compiler.py` / `triton_ascend.cc` 等调试开关）自然出现在工作树里；它们不挂在代码轨上，
  所以 `pack-load` 这类 PR 分支是"上游 + 纯代码改动"的干净形态。
- **底座本来就推不上去**：`private-commits` 把 `localhack`、`debug` 与整条本地层后代一起锁（不锁祖先，
  否则整条 trunk 历史都算 private）。`origin/localhack` 已经存在，所以日常无感；万一要在远端**首次**创建该分支，
  只能绕开 jj 闸门（`git push origin <commit>:refs/heads/localhack` 后 `jj git fetch`）。
- **仓库级配置的位置容易记错**：本仓库有 `.jj/repo/config-id`（`b6a6840f3d7f267c986a`），repo config 实际是
  `~/.config/jj/repos/b6a6840f3d7f267c986a/config.toml`（`trunk()` 就在这里）；写 `.jj/repo/config.toml` 会被
  静默忽略（实测 `jj config get aliases.mnew` = `Value not found`）。`jj config set --repo` 也指向前者。
- **`AGENTS/**` 一律走文档轨**：装上本工作流后，通用 doc delta 不再手工折叠回底座
  （`AGENTS/LocalTooling.md` §1 与根 `AGENTS.md` §1 的 fold-back 规则已被本流程取代，两处均已加注），
  而是挂在 `localbase()` 上的阶梯节点，`mland` 时逐级沉淀，底座因此无需可改写。
- **`.pi/` 已被 `.git/info/exclude` 忽略**（见 `AGENTS/LocalTooling.md` §3），因此需求账本、review 收据
  不会混进代码轨，也不会被 `private` 误判。
- `localhack` 的 change id 是 divergent（`ukxrlwzt/0` 等多份副本），别名只按**书签**寻址（`localhack`），
  不要用 change id 裸写。

---

## 5. 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| `mmega: @ 还有未归位改动` | `@` 自己的 diff 不为空时重建节点会把改动落在孤儿提交里。先 `jj mput <pr>` / `jj mback`，或 `jj new` 把改动留在旧提交。 |
| `mback: 当前 @ 不在阶梯节点之上` | `@` 的父不是 $H_k$（例如停在代码轨上）。先 `jj mgo <P_k>`。 |
| `mgo: ... 没有配对的阶梯节点` | 传入的 rev 既不在 `localbase()::` 里、也没有 `merge` 子提交；确认它是某个 PR 的代码提交。 |
| 阶梯节点出现冲突 | 这是设计内的：$H_{k+1}$ = merge($H_k$, $P_{k+1}$) 与本地层 hack 改同一文件时会冲突。`jj mgo <P_{k+1}>` 进去解，然后 `jj mback`。 |
| `mland: 阶梯节点与本地层在文档上冲突了` | 另一笔 PR 先把同一行文档沉淀了。`mland` 此时已把阶梯 rebase 到新本地层、但**没有**前移 `localbase` 书签、也没删 `hack-<pr>`，所以可以直接 `jj mgo <P_1>` → 按要留下的内容改文件 → `jj mback` → 重跑 `jj mland <pr>`。别手动删阶梯：解完冲突后沉淀才会按级正确落盘。 |
| `mland` 说 `localbase() 必须指向一个有本地书签的提交` | `localbase()` 解析不到本地书签（远端书签 `main@origin` 不行，`jj bookmark set main@origin` 直接报 `Failed to parse bookmark name`）。 |
| 旧布局残留 `mega-*` 书签 | 直接 `jj mmega`：它先 `bookmark delete 'glob:"mega-*"'` 再重建。 |
| 把文档 `jj squash` 进 `localbase()` 后冒出多节点冲突 + `localbase` 变 divergent | 本地层新增的 `private` 文件改动会与**每个**阶梯节点自己那份同文件改动形成 3-way 冲突（本仓库实测：一次 `squash --into localhack` 改写 25 个后代、10 个 `hack:*` 节点冲突、`localhack` divergent）。文档一律走 `mland` 逐级沉淀，不要 squash 进本地层；已经发生了就 `jj op log` 找到那次 squash、`jj op restore <op>` 回退（`jj undo` 只回一步，而 squash 与后续操作可能已分多步）。 |
| 同名 PR 重开时报 `bookmark ... already exists` | `jj bookmark delete` 只留 `(deleted)` 墓碑，会挡住同名 `bookmark create`；`mnew` 已内置 `jj bookmark forget`。墓碑要顺带删远端分支得再跑 `jj git push --deleted`。 |

---

## 6. 实测事实（jj 0.46.0，别再猜）

1. **`jj new A B` 不会消除冗余父**：即使 A 是 B 的祖先，结果仍是 2 个父。旧的 V 形拓扑因此在
   `mega-<pr> = merge(localbase(), <pr>, hack-<pr>)` 上画出第 3 条边（`localbase()` 旁路）；阶梯拓扑
   干脆不传多余的那个父，于是恒为二叉。
2. **`jj rebase -o` 是 `--onto`（0.46）**，`-A` 是 `--insert-after` 且**替换**父列表；给一个提交换父集合要用
   重复 `-o`：`jj rebase -s <node> -o localbase() -o <pr>`（空格分隔的多个 revset 是语法错误）。
3. **`jj split -r @ -o <rev> <fileset> -m <msg>`**：被选中的改动成为 `<rev>` 的一个新子提交，`@` 保留其余改动。
   `mcommit` 就是靠这个把代码单独抽成 $P_{k+1}$。
4. **`jj absorb --into <revset>`** 在 0.46 存在；`mput --absorb` 用它把改动下沉到引入该行的代码提交。
5. **`private-commits` 只影响 `jj git push`**：对一个被判为 private 的提交做 `squash`/`abandon`/`rebase`
   仍然成功（实测），所以闸门里可以放心写 `localbase()::`；反过来 `armbase()::` 在本仓库会把整棵树判成
   private（本地层也源自 `main-dev@upstream`），别写。
6. **节点不能是 `@`**：所有命令的"`@` 是否干净"检查读的是 `@` 自己的 diff（`jj diff -r @`），一旦把节点
   本身留在 `@` 上（例如把文档 squash 进 `@` 却不新建空提交），下一次 `mmega` 就会误判为"未归位"。
7. **`jj log -T 'change_id'` 不输出结尾换行**：把它塞进 `for`/`tac` 会把多个 id 粘成一行（`mland` 曾因此
   报 `Revset change_id("…") didn't resolve`）-> 循环里的模板必须写 `change_id ++ "\n"`。同理
   `jj log -r <空 revset>` 退出码是 0，不能用退出码判断"有没有命中"。
8. **revset 不支持字符串拼接**：`bookmarks(exact:"hack-" ++ pr)` 解析失败，所以没有按 PR 名参数化的
   `hack_chain(pr)` 别名；调用点直接写
   `(::bookmarks(exact:"hack-<pr>")) & merges() & ~(::armbase() | ::localbase())`。
9. **`jj bookmark list` 的模板没有 `commit_id` / `local_bookmarks` 关键字**（`jj log` 里 `commit_id` 可用）；
   取本地书签名用 `jj log -r 'localbase() & bookmarks()' -T 'bookmarks'` 再滤掉 `@…`。
10. **`roots(A..B)` 在 B 是 A 的祖先时为空**（B 落后 A）：所以 `mland` 的"本地层对齐 trunk"按
    `[Ahead]` / `[Behind]` / `[Diverged]` 三个互斥分支处理（已在上面 / 书签快进 / `rebase -o trunk()`），
    不要用单一 `roots` 判定三种情形。
11. **`fileset` 的 `glob:"**/X"` 匹配仓库根**：`**/AGENTS.md`、`**/AGENTS/**` 同时覆盖根层，无需再写根形式；
    `fileset-aliases` 用裸名字引用（`jj diff --name-only '~private'`）。
12. **`jj squash --from X --into Y` 在脚本里必须带 `-u` 或 `-m`**：两边都有描述时会开编辑器，无编辑器直接失败。
13. **不要用 `jj squash --from <merge 节点>` 做逐步沉淀**：它会先把该节点 rebase 到目标上再算 diff，于是
    "本步文档"变成"累积文档"，与外一层落的提交叠加就冲突。`mland` 的做法是
    `rebase -s H1 -o localbase() -o P1`（让 3-way 合并处理好本地层的新文档）之后按节点 `jj restore --from` 取累积内容，
    每级仍然只产生本步的 delta。
14. **`X::` 是后代、`::X` 是祖先**：区分"代码提交"与"本地层提交"时别写反——`remote_bookmarks()::` 会把
    整个本地层的后代都算进去（实测列出 `@`），要的是 `::remote_bookmarks()`。
15. **阶梯节点的代码父只能按结构识别**：`parents(H_k) ~ merges()` 去掉阶梯父（$H_{k-1}$ 是 merge，$P_k$ 不是），
    但 $H_1$ 的两个父都不是 merge；而本地层被平移到新 trunk 之后**代码提交反而成了 `localbase()` 的祖先**
    （本地层复用了它的树），所以"不在 `::localbase()` 里"这条判据会失效。`paired_code(h)` 因此再接一步
    排除"只存在于本地层、没上过远端"的提交：
    `(parents(h) ~ merges()) ~ ((::localbase()) ~ (::remote_bookmarks()))`。
16. **`mland` 的 $P_1$ 必须在本地层对齐之前取**：对齐会重写本地层根，之后 `parents(H_1) ~ (::localbase())`
    已经分辨不出哪个父是代码提交；对齐前取、对齐后连同 `-o P1` 一起 rebase，冲突中止时配对仍然完整，
    `mgo`/`mback` 才找得到 $P_1$。

---

## 7. 从旧 V 形拓扑迁移

旧的 `mega-<pr> = merge(localbase(), <pr>, hack-<pr>)` 布局转成阶梯：

```bash
# 1) 逐级建阶梯：P 链不动，每级 merge 上一层节点与该级代码，再把该级 doc 提交的 delta 塞进去
jj new 'localbase()' 'P1' -m "hack: <第一步信息>"
jj squash -u --from <D1> --into @ 'private'
jj new @ 'P2' -m "hack: <第二步信息>"
jj squash -u --from <D2> --into @ 'private'
# 2) 把文档轨书签指到阶梯尖端，卸掉旧节点书签
jj bookmark set hack-<pr> -r @
jj bookmark delete mega-<pr>
jj mmega
```

要点：doc 提交 $D_k$ 是相对它**自己的父**的 delta（`squash --from` 是 3-way），所以不会把旧基线里的
文档内容一起搬过来——`jj restore --from <别的分支提交>` 才会（它搬整份文件内容）。
