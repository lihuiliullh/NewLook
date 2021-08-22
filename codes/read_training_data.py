args.data_path = "../data/FB15k"
tasks = "1c.2c.3c.2i.3i.ic.ci.2u.uc"


train_ans = dict()
valid_ans = dict()
valid_ans_hard = dict()
test_ans = dict()
test_ans_hard = dict()

if '1c' in tasks:
    with open('%s/train_triples_1c.pkl' % args.data_path, 'rb') as handle:
        train_triples = pickle.load(handle)
    with open('%s/train_ans_1c.pkl' % args.data_path, 'rb') as handle:
        train_ans_1 = pickle.load(handle)
    with open('%s/valid_triples_1c.pkl' % args.data_path, 'rb') as handle:
        valid_triples = pickle.load(handle)
    with open('%s/valid_ans_1c.pkl' % args.data_path, 'rb') as handle:
        valid_ans_1 = pickle.load(handle)
    with open('%s/valid_ans_1c_hard.pkl' % args.data_path, 'rb') as handle:
        valid_ans_1_hard = pickle.load(handle)
    with open('%s/test_triples_1c.pkl' % args.data_path, 'rb') as handle:
        test_triples = pickle.load(handle)
    with open('%s/test_ans_1c.pkl' % args.data_path, 'rb') as handle:
        test_ans_1 = pickle.load(handle)
    with open('%s/test_ans_1c_hard.pkl' % args.data_path, 'rb') as handle:
        test_ans_1_hard = pickle.load(handle)
    train_ans.update(train_ans_1)
    valid_ans.update(valid_ans_1)
    valid_ans_hard.update(valid_ans_1_hard)
    test_ans.update(test_ans_1)
    test_ans_hard.update(test_ans_1_hard)

if '2c' in tasks:
    with open('%s/train_triples_2c.pkl' % args.data_path, 'rb') as handle:
        train_triples_2 = pickle.load(handle)
    with open('%s/train_ans_2c.pkl' % args.data_path, 'rb') as handle:
        train_ans_2 = pickle.load(handle)
    with open('%s/valid_triples_2c.pkl' % args.data_path, 'rb') as handle:
        valid_triples_2 = pickle.load(handle)
    with open('%s/valid_ans_2c.pkl' % args.data_path, 'rb') as handle:
        valid_ans_2 = pickle.load(handle)
    with open('%s/valid_ans_2c_hard.pkl' % args.data_path, 'rb') as handle:
        valid_ans_2_hard = pickle.load(handle)
    with open('%s/test_triples_2c.pkl' % args.data_path, 'rb') as handle:
        test_triples_2 = pickle.load(handle)
    with open('%s/test_ans_2c.pkl' % args.data_path, 'rb') as handle:
        test_ans_2 = pickle.load(handle)
    with open('%s/test_ans_2c_hard.pkl' % args.data_path, 'rb') as handle:
        test_ans_2_hard = pickle.load(handle)
    train_ans.update(train_ans_2)
    valid_ans.update(valid_ans_2)
    valid_ans_hard.update(valid_ans_2_hard)
    test_ans.update(test_ans_2)
    test_ans_hard.update(test_ans_2_hard)

if '3c' in tasks:
    with open('%s/train_triples_3c.pkl' % args.data_path, 'rb') as handle:
        train_triples_3 = pickle.load(handle)
    with open('%s/train_ans_3c.pkl' % args.data_path, 'rb') as handle:
        train_ans_3 = pickle.load(handle)
    with open('%s/valid_triples_3c.pkl' % args.data_path, 'rb') as handle:
        valid_triples_3 = pickle.load(handle)
    with open('%s/valid_ans_3c.pkl' % args.data_path, 'rb') as handle:
        valid_ans_3 = pickle.load(handle)
    with open('%s/valid_ans_3c_hard.pkl' % args.data_path, 'rb') as handle:
        valid_ans_3_hard = pickle.load(handle)
    with open('%s/test_triples_3c.pkl' % args.data_path, 'rb') as handle:
        test_triples_3 = pickle.load(handle)
    with open('%s/test_ans_3c.pkl' % args.data_path, 'rb') as handle:
        test_ans_3 = pickle.load(handle)
    with open('%s/test_ans_3c_hard.pkl' % args.data_path, 'rb') as handle:
        test_ans_3_hard = pickle.load(handle)
    train_ans.update(train_ans_3)
    valid_ans.update(valid_ans_3)
    valid_ans_hard.update(valid_ans_3_hard)
    test_ans.update(test_ans_3)
    test_ans_hard.update(test_ans_3_hard)

if '2i' in tasks:
    with open('%s/train_triples_2i.pkl' % args.data_path, 'rb') as handle:
        train_triples_2i = pickle.load(handle)
    with open('%s/train_ans_2i.pkl' % args.data_path, 'rb') as handle:
        train_ans_2i = pickle.load(handle)
    with open('%s/valid_triples_2i.pkl' % args.data_path, 'rb') as handle:
        valid_triples_2i = pickle.load(handle)
    with open('%s/valid_ans_2i.pkl' % args.data_path, 'rb') as handle:
        valid_ans_2i = pickle.load(handle)
    with open('%s/valid_ans_2i_hard.pkl' % args.data_path, 'rb') as handle:
        valid_ans_2i_hard = pickle.load(handle)
    with open('%s/test_triples_2i.pkl' % args.data_path, 'rb') as handle:
        test_triples_2i = pickle.load(handle)
    with open('%s/test_ans_2i.pkl' % args.data_path, 'rb') as handle:
        test_ans_2i = pickle.load(handle)
    with open('%s/test_ans_2i_hard.pkl' % args.data_path, 'rb') as handle:
        test_ans_2i_hard = pickle.load(handle)
    train_ans.update(train_ans_2i)
    valid_ans.update(valid_ans_2i)
    valid_ans_hard.update(valid_ans_2i_hard)
    test_ans.update(test_ans_2i)
    test_ans_hard.update(test_ans_2i_hard)

if '3i' in tasks:
    with open('%s/train_triples_3i.pkl' % args.data_path, 'rb') as handle:
        train_triples_3i = pickle.load(handle)
    with open('%s/train_ans_3i.pkl' % args.data_path, 'rb') as handle:
        train_ans_3i = pickle.load(handle)
    with open('%s/valid_triples_3i.pkl' % args.data_path, 'rb') as handle:
        valid_triples_3i = pickle.load(handle)
    with open('%s/valid_ans_3i.pkl' % args.data_path, 'rb') as handle:
        valid_ans_3i = pickle.load(handle)
    with open('%s/valid_ans_3i_hard.pkl' % args.data_path, 'rb') as handle:
        valid_ans_3i_hard = pickle.load(handle)
    with open('%s/test_triples_3i.pkl' % args.data_path, 'rb') as handle:
        test_triples_3i = pickle.load(handle)
    with open('%s/test_ans_3i.pkl' % args.data_path, 'rb') as handle:
        test_ans_3i = pickle.load(handle)
    with open('%s/test_ans_3i_hard.pkl' % args.data_path, 'rb') as handle:
        test_ans_3i_hard = pickle.load(handle)
    train_ans.update(train_ans_3i)
    valid_ans.update(valid_ans_3i)
    valid_ans_hard.update(valid_ans_3i_hard)
    test_ans.update(test_ans_3i)
    test_ans_hard.update(test_ans_3i_hard)

if 'ci' in tasks:
    with open('%s/valid_triples_ci.pkl' % args.data_path, 'rb') as handle:
        valid_triples_ci = pickle.load(handle)
    with open('%s/valid_ans_ci.pkl' % args.data_path, 'rb') as handle:
        valid_ans_ci = pickle.load(handle)
    with open('%s/valid_ans_ci_hard.pkl' % args.data_path, 'rb') as handle:
        valid_ans_ci_hard = pickle.load(handle)
    with open('%s/test_triples_ci.pkl' % args.data_path, 'rb') as handle:
        test_triples_ci = pickle.load(handle)
    with open('%s/test_ans_ci.pkl' % args.data_path, 'rb') as handle:
        test_ans_ci = pickle.load(handle)
    with open('%s/test_ans_ci_hard.pkl' % args.data_path, 'rb') as handle:
        test_ans_ci_hard = pickle.load(handle)
    valid_ans.update(valid_ans_ci)
    valid_ans_hard.update(valid_ans_ci_hard)
    test_ans.update(test_ans_ci)
    test_ans_hard.update(test_ans_ci_hard)

if 'ic' in tasks:
    with open('%s/valid_triples_ic.pkl' % args.data_path, 'rb') as handle:
        valid_triples_ic = pickle.load(handle)
    with open('%s/valid_ans_ic.pkl' % args.data_path, 'rb') as handle:
        valid_ans_ic = pickle.load(handle)
    with open('%s/valid_ans_ic_hard.pkl' % args.data_path, 'rb') as handle:
        valid_ans_ic_hard = pickle.load(handle)
    with open('%s/test_triples_ic.pkl' % args.data_path, 'rb') as handle:
        test_triples_ic = pickle.load(handle)
    with open('%s/test_ans_ic.pkl' % args.data_path, 'rb') as handle:
        test_ans_ic = pickle.load(handle)
    with open('%s/test_ans_ic_hard.pkl' % args.data_path, 'rb') as handle:
        test_ans_ic_hard = pickle.load(handle)
    valid_ans.update(valid_ans_ic)
    valid_ans_hard.update(valid_ans_ic_hard)
    test_ans.update(test_ans_ic)
    test_ans_hard.update(test_ans_ic_hard)

if 'uc' in tasks:
    with open('%s/valid_triples_uc.pkl' % args.data_path, 'rb') as handle:
        valid_triples_uc = pickle.load(handle)
    with open('%s/valid_ans_uc.pkl' % args.data_path, 'rb') as handle:
        valid_ans_uc = pickle.load(handle)
    with open('%s/valid_ans_uc_hard.pkl' % args.data_path, 'rb') as handle:
        valid_ans_uc_hard = pickle.load(handle)
    with open('%s/test_triples_uc.pkl' % args.data_path, 'rb') as handle:
        test_triples_uc = pickle.load(handle)
    with open('%s/test_ans_uc.pkl' % args.data_path, 'rb') as handle:
        test_ans_uc = pickle.load(handle)
    with open('%s/test_ans_uc_hard.pkl' % args.data_path, 'rb') as handle:
        test_ans_uc_hard = pickle.load(handle)
    valid_ans.update(valid_ans_uc)
    valid_ans_hard.update(valid_ans_uc_hard)
    test_ans.update(test_ans_uc)
    test_ans_hard.update(test_ans_uc_hard)

if '2u' in tasks:
    with open('%s/valid_triples_2u.pkl' % args.data_path, 'rb') as handle:
        valid_triples_2u = pickle.load(handle)
    with open('%s/valid_ans_2u.pkl' % args.data_path, 'rb') as handle:
        valid_ans_2u = pickle.load(handle)
    with open('%s/valid_ans_2u_hard.pkl' % args.data_path, 'rb') as handle:
        valid_ans_2u_hard = pickle.load(handle)
    with open('%s/test_triples_2u.pkl' % args.data_path, 'rb') as handle:
        test_triples_2u = pickle.load(handle)
    with open('%s/test_ans_2u.pkl' % args.data_path, 'rb') as handle:
        test_ans_2u = pickle.load(handle)
    with open('%s/test_ans_2u_hard.pkl' % args.data_path, 'rb') as handle:
        test_ans_2u_hard = pickle.load(handle)
    valid_ans.update(valid_ans_2u)
    valid_ans_hard.update(valid_ans_2u_hard)
    test_ans.update(test_ans_2u)
    test_ans_hard.update(test_ans_2u_hard)