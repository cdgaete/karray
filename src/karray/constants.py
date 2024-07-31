


css_style = '''<style>

.details {
    user-select: none;
}

.details > summary {
    display: flex;
    cursor: pointer;
    position: relative;
}

.details > summary .span.icon {
    width: 24px;
    height: 24px;
    transition: all 0.3s;
    margin-left: auto;
}

.details[open] > summary.summary ::-webkit-details-marker {
    display: none;
}

.details[open] > summary .span.icon {
    transform: rotate(180deg);
}

/* Tooltip styles */
.tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted black;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 165px;
    background-color: black;
    color: #fff;
    text-align: center;
    border-radius: 4px;
    padding: 2px 0;
    position: absolute;
    z-index: 1;
    font-size: 11px;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
}

.tooltip .tooltiptext::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -8px;
    border-width: 8px;
    border-style: solid;
    border-color: #fff transparent transparent transparent;
}

.tooltip-top {
    bottom: 90%;
    margin-left: -40px;
}
</style>'''