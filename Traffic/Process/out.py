"""
.. module:: out

out
*************

:Description: out

    Functions extracted from Generate Dataset no needed any more

:Authors: bejar
    

:Version: 

:Created on: 14/02/2017 14:45 

"""

__author__ = 'bejar'



# def generate_dataset_PCA(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=100, method='one', cpatt=None, reshape=False):
#     """
#     Generates a training and test datasets from the days in the parameters
#     z_factor is the zoom factor to rescale the images
#     :param ldaysTr:
#     :param ldaysTs:
#     :param z_factor:
#     :param PCA:
#     :param method:
#     :return:
#
#     """
#     # -------------------- Train Set ------------------
#     ldataTr = []
#     llabelsTr = []
#
#     for day in ldaysTr:
#         if method == 'one':
#             dataset = generate_classification_dataset_one(day, cpatt=cpatt)
#         else:
#             dataset = generate_classification_dataset_two(day, cpatt=cpatt)
#         for t in dataset:
#             for cam, l, _, _ in dataset[t]:
#                 if l != 0 and l != 6:
#                     image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
#                     if np.sum(image == 254) < 100000: # This avoids the "not Available data" image
#                         del image
#                         im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
#                         data = np.asarray(im)
#                         data = data[5:235, 5:315, :].astype('float32')
#                         data /= 255.0
#                         if z_factor is not None:
#                             data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
#                                               zoom(data[:, :, 2], z_factor)))
#                         if reshape:
#                             data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
#                         ldataTr.append(data)
#                         llabelsTr.append(l)
#
#     # ------------- Test Set ------------------
#     ldataTs = []
#     llabelsTs = []
#
#     for day in ldaysTs:
#         if method == 'one':
#             dataset = generate_classification_dataset_one(day, cpatt=cpatt)
#         else:
#             dataset = generate_classification_dataset_two(day, cpatt=cpatt)
#         for t in dataset:
#             for cam, l, _, _ in dataset[t]:
#                 # print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
#                 if l != 0 and l != 6:
#                     image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
#                     if np.sum(image == 254) < 100000: # This avoids the "not Available data" image
#                         del image
#                         im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
#                         data = np.asarray(im)
#                         data = data[5:235, 5:315, :].astype('float32')
#                         data /= 255.0
#                         if z_factor is not None:
#                             data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
#                                               zoom(data[:, :, 2], z_factor)))
#                         if reshape:
#                             data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
#                         ldataTs.append(data)
#                         llabelsTs.append(l)
#
#     if reshape or z_factor is not None:
#         del data
#
#     print(Counter(llabelsTr))
#     print(Counter(llabelsTs))
#
#     X_train = np.array(ldataTr)
#     del ldataTr
#     X_test = np.array(ldataTs)
#     del ldataTs
#
#     if PCA:
#         pca = IncrementalPCA(n_components=ncomp)
#         pca.fit(X_train)
#         print(np.sum(pca.explained_variance_ratio_[:ncomp]))
#         X_train = pca.transform(X_train)
#         X_test = pca.transform(X_test)
#
#     y_train = llabelsTr
#     y_test = llabelsTs
#     print(X_train.shape, X_test.shape)
#
#     return X_train, y_train, X_test, y_test


# def save_daily_dataset(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=100, method='one', cpatt=None, reshape=False):
#     """
#     Computes the PCA transformation using the days in ldaysTr
#     Generates and save datasets from the days in the ldaysTs
#     z_factor is the zoom factor to rescale the images
#     :param trdays:
#     :param tsdays:
#     :return:
#     """
#
#     # -------------------- Train Set ------------------
#     ldataTr = []
#     llabelsTr = []
#
#     for day in ldaysTr:
#         if method == 'one':
#             dataset = generate_classification_dataset_one(day, cpatt=cpatt)
#         else:
#             dataset = generate_classification_dataset_two(day, cpatt=cpatt)
#         for t in dataset:
#             for cam, l, _, _ in dataset[t]:
#                 if l != 0 and l != 6:
#                     image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
#                     if np.sum(image == 254) < 100000:
#                         del image
#                         im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
#                         data = np.asarray(im)
#                         data = data[5:235, 5:315, :].astype('float32')
#                         data /= 255.0
#                         if z_factor is not None:
#                             data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
#                                               zoom(data[:, :, 2], z_factor)))
#                         if reshape:
#                             data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
#
#                         ldataTr.append(data)
#                         llabelsTr.append(l)
#
#     print(Counter(llabelsTr))
#     X_train = np.array(ldataTr)
#     pca = IncrementalPCA(n_components=ncomp)
#     pca.fit(X_train)
#     print(np.sum(pca.explained_variance_ratio_[:ncomp]))
#     del X_train
#
#     # ------------- Test Set ------------------
#     for day in ldaysTs:
#         ldataTs = []
#         llabelsTs = []
#         if method == 'one':
#             dataset = generate_classification_dataset_one(day, cpatt=cpatt)
#         else:
#             dataset = generate_classification_dataset_two(day, cpatt=cpatt)
#         for t in dataset:
#             for cam, l, _, _ in dataset[t]:
#                 # print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
#                 if l != 0 and l != 6:
#                     image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
#                     if np.sum(image == 254) < 100000:
#                         del image
#                         im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
#                         data = np.asarray(im)
#                         data = data[5:235, 5:315, :].astype('float32')
#                         data /= 255.0
#                         if z_factor is not None:
#                             data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
#                                               zoom(data[:, :, 2], z_factor)))
#                         if reshape:
#                             data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
#                         ldataTs.append(data)
#                         llabelsTs.append(l)
#         X_test = pca.transform(np.array(ldataTs))
#         y_test = llabelsTs
#         print(Counter(llabelsTs))
#         np.save(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp), X_test)
#         np.save(dataset_path + 'labels-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp), np.array(y_test))


# def generate_rebalanced_dataset(ldaysTr, ndays, z_factor, PCA=True, ncomp=100):
#     """
#     Generates a training dataset with a rebalance of the classes using a specific number of days of
#     the input files for the training dataset
#
#     :param ldaysTr:
#     :param z_factor:
#     :param PCA:
#     :param ncomp:
#     :return:
#     """
#     ldata = []
#     y_train = []
#
#     for cl, nd in ndays:
#         for i in range(nd):
#             day = ldaysTr[i]
#             data = np.load(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp))
#             labels = np.load(dataset_path + 'labels-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp))
#             ldata.append(data[labels==cl,:])
#             y_train.extend(labels[labels==cl])
#     X_train = np.concatenate(ldata)
#     print(X_train.shape)
#     print(Counter(y_train))
#     np.save(dataset_path + 'data-RB-Z%0.2f-C%d.npy' % (z_factor, ncomp), X_train)
#     np.save(dataset_path + 'labels-RB-Z%0.2f-C%d.npy' % (z_factor, ncomp), np.array(y_train))


# def generate_rebalanced_data_day(day, z_factor, pclasses):
#     """
#     Generates a rebalanced dataset using the probability of the examples indicated in the parameter pclasses
#
#     :param day:
#     :param z_factor:
#     :param nclasses:
#     :return:
#     """
#     ddata = {}
#     for c in pclasses:
#         if os.path.exists(process_path + 'data-D%s-Z%0.2f-L%d.npy' % (day, z_factor, c)):
#             ddata[c] = np.load(process_path + 'data-D%s-Z%0.2f-L%d.npy' % (day, z_factor, c))
#
#     ldata = []
#     llabels = []
#     for c in pclasses:
#         if c in ddata:
#             nex = np.array(range(ddata[c].shape[0]))
#             shuffle(nex)
#             nsel = int(ddata[c].shape[0] * pclasses[c])
#             sel = nex[0:nsel]
#             ldata.append(ddata[c][sel])
#             llabels.append(np.zeros(nsel)+c)
#
#     np.save(dataset_path + 'rdata-D%s-Z%0.2f.npy' % (day, z_factor), np.concatenate(ldata))
#     np.save(dataset_path + 'rlabels-D%s-Z%0.2f.npy' % (day, z_factor), np.concatenate(llabels))



# def load_generated_dataset(datapath, ldaysTr, z_factor):
#     """
#     Load the  dataset files for a list of days
#
#     :param ldaysTr:
#     :param ldaysTs:
#     :param z_factor:
#     :return:
#     """
#     ldata = []
#     y_train = []
#     for day in ldaysTr:
#         data = np.load(datapath + 'data-D%s-Z%0.2f.npy' % (day, z_factor))
#         ldata.append(data)
#         y_train.extend(np.load(datapath + 'labels-D%s-Z%0.2f.npy' % (day, z_factor)))
#     X_train = np.concatenate(ldata)
#
#     return X_train, y_train


# def generate_splitted_data_day(day, z_factor, method='two', log=False):
#     """
#     Generates a raw dataset for a day with a zoom factor splitted in as many files as classes
#     :param z_factor:
#     :return:
#     """
#     ldata = []
#     llabels = []
#     if method == 'one':
#         dataset = generate_classification_dataset_one(day)
#     else:
#         dataset = generate_classification_dataset_two(day)
#     for t in dataset:
#         for cam, l, _, _ in dataset[t]:
#             if l != 0 and l != 6:
#                 if log:
#                     print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
#                 image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
#                 if np.sum(image == 254) < 100000:
#                     del image
#                     im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
#
#                     data = np.asarray(im)
#                     data = data[5:235, 5:315, :].astype('float32')
#                     data /= 255.0
#                     if z_factor is not None:
#                         data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
#                                           zoom(data[:, :, 2], z_factor)))
#
#                     ldata.append(data)
#                     llabels.append(l)
#
#     llabels = np.array(llabels) -1  # labels in range [0,max classes-1]
#     data = np.array(ldata)
#     for l in np.unique(llabels):
#         sel = llabels == l
#         np.save(process_path + 'data-D%s-Z%0.2f-L%d.npy' % (day, z_factor, l), data[sel])

