import asyncio


async def async_sleep(num):
    import time
    time.sleep(num)


async def coro(tag):
    print(">", tag)
    await asyncio.sleep(3)
    print("<", tag)
    return tag


async def coro2(tag):
    print(">", tag)
    await asyncio.sleep(tag)
    print("<", tag)
    return tag

async def coro3():
    print('yey')
    return


loop = asyncio.get_event_loop()

# group1 = asyncio.gather(*[coro("group 1.{}".format(i)) for i in range(1, 6)])
# group2 = asyncio.gather(*[coro("group 2.{}".format(i)) for i in range(1, 4)])
# group3 = asyncio.gather(*[coro2() for i in range(1, 10)])

# all_groups = asyncio.gather(group1, group2, group3)
# all_groups = asyncio.gather(coro2(5), coro2(1))

async def gather():
    await coro2(5)
    await coro2(1)
    return

# coro3()

results = loop.run_until_complete(gather())

loop.close()

# pprint(results)
