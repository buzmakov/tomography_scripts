import astra

def gpu_fp(pg, vg, v):
    v_id = astra.data2d.create('-vol', vg, v)
    rt_id = astra.data2d.create('-sino', pg)
    fp_cfg = astra.astra_dict('FP_CUDA')
    fp_cfg['VolumeDataId'] = v_id
    fp_cfg['ProjectionDataId'] = rt_id
    fp_id = astra.algorithm.create(fp_cfg)
    astra.algorithm.run(fp_id)
    out = astra.data2d.get(rt_id)
    
    astra.algorithm.delete(fp_id)
    astra.data2d.delete(rt_id)
    astra.data2d.delete(v_id)
    return out

def gpu_bp(pg, vg, rt, supersampling=1):
    v_id = astra.data2d.create('-vol', vg)
    rt_id = astra.data2d.create('-sino', pg, data=rt)
    bp_cfg = astra.astra_dict('BP_CUDA')
    bp_cfg['ReconstructionDataId'] = v_id
    bp_cfg['ProjectionDataId'] = rt_id
    bp_id = astra.algorithm.create(bp_cfg)
    astra.algorithm.run(bp_id)
    out = astra.data2d.get(v_id)
    
    astra.algorithm.delete(bp_id)
    astra.data2d.delete(rt_id)
    astra.data2d.delete(v_id)
    return out

def gpu_fbp(pg, vg, rt):
    rt_id = astra.data2d.create('-sino', pg, data=rt)
    v_id = astra.data2d.create('-vol', vg)
    fbp_cfg = astra.astra_dict('FBP_CUDA')
    fbp_cfg['ReconstructionDataId'] = v_id
    fbp_cfg['ProjectionDataId'] = rt_id
    fbp_cfg['option'] = {}
    fbp_cfg['option']['ShortScan'] = True
    #fbp_cfg['FilterType'] = 'none'
    fbp_id = astra.algorithm.create(fbp_cfg)
    astra.algorithm.run(fbp_id, 100)
    out = astra.data2d.get(v_id)

    astra.algorithm.delete(fbp_id)
    astra.data2d.delete(rt_id)
    astra.data2d.delete(v_id)
    return out

def gpu_sirt(pg, vg, rt, n_iters=100, x0=0.0):
    rt_id = astra.data2d.create('-sino', pg, data=rt)
    v_id = astra.data2d.create('-vol', vg, x0)
    sirt_cfg = astra.astra_dict('SIRT_CUDA')
    sirt_cfg['ReconstructionDataId'] = v_id
    sirt_cfg['ProjectionDataId'] = rt_id
    #sirt_cfg['option'] = {}
    #sirt_cfg['option']['MinConstraint'] = 0
    sirt_id = astra.algorithm.create(sirt_cfg)
    astra.algorithm.run(sirt_id, n_iters)
    out = astra.data2d.get(v_id)

    astra.algorithm.delete(sirt_id)
    astra.data2d.delete(rt_id)
    astra.data2d.delete(v_id)
    return out

def gpu_cgls(pg, vg, rt, n_iters=100, x0=0.0):
    rt_id = astra.data2d.create('-sino', pg, data=rt)
    v_id = astra.data2d.create('-vol', vg, x0)
    sirt_cfg = astra.astra_dict('CGLS_CUDA')
    sirt_cfg['ReconstructionDataId'] = v_id
    sirt_cfg['ProjectionDataId'] = rt_id
    #sirt_cfg['option'] = {}
    #sirt_cfg['option']['MinConstraint'] = 0
    sirt_id = astra.algorithm.create(sirt_cfg)
    astra.algorithm.run(sirt_id, n_iters)
    out = astra.data2d.get(v_id)

    astra.algorithm.delete(sirt_id)
    astra.data2d.delete(rt_id)
    astra.data2d.delete(v_id)
    return out
